import six
import pickle
from pydoc import locate
from itertools import chain
from functools import partial
from collections import namedtuple

import numpy as np
import tensorflow as tf

from seqmodel import util
from seqmodel import dstruct
from seqmodel import graph as tfg
from seqmodel import cells as tfcell
from seqmodel import model as _sqm


__all__ = [
    'UnigramSeqModel', 'UnigramSeqModelH', 'BackwardSeqModel']

NONE = tf.no_op()
tfdense = tf.layers.dense
_nax_ = tf.newaxis


def sample_normal(mu, scale):
    epsilon = tf.random_normal(tf.shape(mu))
    sample = mu + tf.multiply(scale, epsilon)
    return sample


def log_sum_exp(x, axis=-1, keep_dims=False):
    a = tf.reduce_max(x, axis, keep_dims=True)
    out = a + tf.log(tf.reduce_sum(tf.exp(x - a), axis, keep_dims=True))
    if keep_dims:
        return out
    else:
        return tf.squeeze(out, [axis])


def kl_normal(mu0, scale0, mu1, scale1):
    # from tensorflow repo
    one = tf.constant(1, dtype=tf.float32)
    two = tf.constant(2, dtype=tf.float32)
    half = tf.constant(0.5, dtype=tf.float32)
    s_a_squared = tf.square(scale0)
    s_b_squared = tf.square(scale1)
    ratio = s_a_squared / s_b_squared
    return (tf.square(mu0 - mu1) / (two * s_b_squared) +
            half * (ratio - one - tf.log(ratio)))


def kl_mvn_diag(mu0, diag_scale0, mu1, diag_scale1):
    return tf.reduce_sum(kl_normal(mu0, diag_scale0, mu1, diag_scale1), axis=-1)


def log_pdf_normal(x, mu, scale):
    var = scale ** 2
    return -0.5 * (np.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var)


def log_pdf_mvn_diag(x, mu, scale):
    return tf.reduce_sum(log_pdf_normal(x, mu, scale), axis=-1)


class DiagGaussianMixture(object):

    def __init__(
            self, n_components=None, dimensions=None, trainable=True, sk_gmm_path=None,
            means=None, scales=None, weights=None,
            activation_mean=None, activation_scale=None):
        init_weights, init_means, init_scales = None, None, None
        self._K = n_components
        self._D = dimensions
        if means is not None and scales is not None and weights is not None:
            self._means = means
            self._scales = scales
            self._weights = weights
            self._K = weights.shape[-1]
            self._D = means.shape[-1]
        else:
            if sk_gmm_path is not None:
                with open(sk_gmm_path, mode='rb') as f:
                    sklearn_gmm = pickle.load(f)
                self._K = sklearn_gmm.n_components
                self._D = sklearn_gmm.means_.shape[-1]
                init_weights = sklearn_gmm.weights_
                init_means = sklearn_gmm.means_
                init_scales = np.sqrt(sklearn_gmm.covariances_)
            with tf.variable_scope('diag_gm') as scope:
                self._weights, self._means, self._scales = \
                    DiagGaussianMixture.create_vars(
                        self.K, self.D, trainable, init_weights, init_means, init_scales,
                        activation_mean=activation_mean,
                        activation_scale=activation_scale)
                self._scope = scope

    def log_pdf_k(self, x, k):
        if len(x.shape) == 2 and len(self._means.shape) == 2:
            mean_k = self._means[tf.newaxis, k, :]  # expand batch axis
            scale_k = self._scales[tf.newaxis, k, :]
        elif len(x.shape) == 1 and len(self._means.shape) == 2:
            mean_k = self._means[k, :]
            scale_k = self._scales[k, :]
        return log_pdf_mvn_diag(x, mean_k, scale_k)

    def log_pdf(self, x):
        weights, means, scales = self._weights, self._means, self._scales
        if len(x.shape) == 2 and len(self._means.shape) == 2:
            x = x[:, tf.newaxis, :]
            means = self._means[tf.newaxis, :, :]
            scales = self._scales[tf.newaxis, :, :]
            weights = self._weights[tf.newaxis, :]
        elif len(x.shape) == 1 and len(self._means.shape) == 2:
            x = x[tf.newaxis, :]
        log_pdf_K = log_pdf_mvn_diag(x, means, scales)
        return log_sum_exp(log_pdf_K + tf.log(weights), axis=-1)

    @property
    def K(self):
        return self._K

    @property
    def D(self):
        return self._D

    @staticmethod
    def create_vars(
            num_components, dimensions, trainable=True,
            init_weights=None, init_means=None, init_scales=None,
            activation_mean=None, activation_scale=None,
            scope=None):
        shape = (num_components, dimensions)
        with tf.variable_scope(scope or 'gm') as scope:
            weights = tfg.create_tensor(
                (num_components, ), trainable=trainable, init=init_weights,
                name='weights')
            means = tfg.create_tensor(
                shape, trainable=trainable, init=init_means, name='means')
            if activation_mean is not None:
                means = activation_mean(means)
            scales = tfg.create_tensor(
                shape, trainable=trainable, init=init_scales, name='scales')
            if activation_scale is not None:
                scales = activation_scale(scales)
        return weights, means, scales


def categorical_graph(
        K, inputs, temperature=1.0, activation=tf.nn.relu, keep_prob=1.0, scope=None):
    input_dim = inputs.shape[-1]
    with tf.variable_scope(scope or 'categorical', reuse=tf.AUTO_REUSE):
        _inputs = inputs
        if keep_prob < 1.0:
            _inputs = tf.nn.dropout(inputs, keep_prob)
        h1 = tfdense(_inputs, input_dim, activation=activation, name='l1')
        if keep_prob < 1.0:
            h1 = tf.nn.dropout(h1, keep_prob)
        h2 = h1 + _inputs
        # h2 = tfdense(h1, input_dim, activation=activation, name='l2')
        if keep_prob < 1.0:
            h2 = tf.nn.dropout(h2, keep_prob)
        logits = tfdense(h2, K, name='logits')
        temp_var = tf.get_variable(
            'gumbel_temperature', dtype=tf.float32, initializer=temperature,
            trainable=False)
        update_temp = tf.assign(temp_var, tf.maximum(0.5, temp_var * 0.99995))
        with tf.control_dependencies([update_temp]):
            gumbel = tf.contrib.distributions.RelaxedOneHotCategorical(
                temp_var, logits=logits)
            sample = gumbel.sample()
            return logits, sample
        # gumbel = tf.contrib.distributions.RelaxedOneHotCategorical(
        #         temperature, logits=logits)
        # sample = gumbel.sample()
        # return logits, sample


def gaussian_graph(
        out_dim, inputs, activation=tf.nn.tanh, scope=None, residual=False,
        mu_activation=None, scale_activation=tf.nn.sigmoid, keep_prob=1.0):
    input_dim = inputs.shape[-1]
    with tf.variable_scope(scope or 'gaussian', reuse=tf.AUTO_REUSE):
        _inputs = inputs
        if keep_prob < 1:
            _inputs = tf.nn.dropout(inputs, keep_prob)
        h1 = tfdense(_inputs, input_dim, activation=activation, name='l1')
        if keep_prob < 1:
            h1 = tf.nn.dropout(h1, keep_prob)
        h2 = tfdense(h1, out_dim * 2, activation=activation, name='l2')
        if keep_prob < 1:
            h2 = tf.nn.dropout(h2, keep_prob)
        mu, scale = tf.split(tfdense(h2, out_dim * 2, name='mu_scale'), 2, axis=-1)
        if mu_activation is not None:
            mu = mu_activation(mu)
        if scale_activation is not None:
            scale = scale_activation(scale)
        if residual:
            mu = mu + inputs
        sample = sample_normal(mu, scale)
    return mu, scale, sample


def gaussian_graph_cat(
        out_dim, inputs, cat, activation=tf.nn.tanh, scope=None, residual=False,
        mu_activation=None, scale_activation=tf.nn.sigmoid, keep_prob=0.0):
    input_dim = inputs.shape[-1]
    with tf.variable_scope(scope or 'gaussian', reuse=tf.AUTO_REUSE):
        _inputs = inputs
        if keep_prob < 1:
            _inputs = tf.nn.dropout(inputs, keep_prob)
        _inputs = tf.concat([cat, _inputs], -1)
        h1 = tfdense(_inputs, input_dim, activation=activation, name='l1')
        if keep_prob < 1:
            h1 = tf.nn.dropout(h1, keep_prob)
        h2 = tfdense(h1, out_dim * 2, activation=activation, name='l2')
        if keep_prob < 1:
            h2 = tf.nn.dropout(h2, keep_prob)
        mu, scale = tf.split(tfdense(h2, out_dim * 2, name='mu_scale'), 2, axis=-1)
        if mu_activation is not None:
            mu = mu_activation(mu)
        if scale_activation is not None:
            scale = scale_activation(scale)
        if residual:
            mu = mu + inputs
        sample = sample_normal(mu, scale)
    return mu, scale, sample


def gaussian_graph_K(
        K, out_dim, inputs, activation=tf.nn.tanh, scope=None,
        mu_activation=tf.nn.tanh, scale_activation=tf.nn.sigmoid):
    means = []
    scales = []
    samples = []
    with tf.variable_scope(scope or 'gmm'):
        for k in range(K):
            mean, scale, sample = gaussian_graph(
                out_dim, inputs, activation=activation, scope=f'gmm_{k}',
                mu_activation=mu_activation, scale_activation=scale_activation)
            means.append(mean)
            scales.append(scale)
            samples.append(sample)
    means = tf.stack(means, 1)
    scales = tf.stack(scales, 1)
    samples = tf.stack(samples, 1)
    return means, scales, samples


def IAF_graph(T, out_dim, inputs, activation=tf.nn.tanh, scope=None):
    with tf.variable_scope(scope or 'iaf', reuse=tf.AUTO_REUSE):
        mu, scale, z = gaussian_graph(
            out_dim, inputs, activation=activation, scope='init',
            scale_activation=tf.nn.sigmoid, residual=False)
        eps = (z - mu) / scale
        neg_log_pdf = tf.log(scale) + 0.5 * (eps**2 + np.log(2*np.pi))
        for t in range(T):
            ms = tfdense(tf.concat([z, inputs], -1), out_dim*2, name=f'iaf_{t}')
            m, s = tf.split(ms, 2, axis=-1)
            scale = tf.nn.sigmoid(s)
            z = scale * z + (1 - scale) * m
            neg_log_pdf += tf.log(scale)
    log_pdf = -tf.reduce_sum(neg_log_pdf, axis=-1)
    return z, log_pdf


class UnigramSeqModel(_sqm.SeqModel):

    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size, reuse_scope, reuse,
            nodes):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        batch_size = self._get_batch_size(lookup)
        input_dim = lookup.shape[-1]
        if opt['out:eval_first_token']:
            lookup = tf.concat((tf.zeros((1, batch_size, input_dim)), lookup), axis=0)
            seq_len += 1
        max_num_tokens = tf.shape(lookup)[0]
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            _reuse = reuse or scope is not None
            cell_ = tfg.create_cells(
                input_size=opt['emb:dim'], wrap_state=True, **cell_opt)
            cell_output_, initial_state_, final_state_ = tfg.create_rnn(
                cell_, lookup, seq_len, initial_state, rnn_fn=opt['rnn:fn'],
                batch_size=batch_size)

            wildcard_lookup = tf.zeros((1, input_dim))
            wildcard_states = tfcell.nested_map(
                lambda state: tf.zeros((1, state.shape[-1])), initial_state_)
            x = wildcard_lookup
            for i, (cell, z) in enumerate(zip(cell_._cells, wildcard_states)):
                with tf.variable_scope(f'rnn/multi_rnn_cell/cell_{i}', reuse=True):
                    x, __ = cell(x, z)
            x = tf.tile(x[tf.newaxis, :, :], [max_num_tokens, batch_size, 1])
            extra_nodes = {'unigram_features': x}
        return cell_, cell_output_, initial_state_, final_state_, extra_nodes

    def _build_loss(
            self, opt, logit, label, weight, seq_weight, nodes, collect_key,
            add_to_collection, inputs=None, **kwargs):
        weight = tf.multiply(weight, seq_weight)
        if opt['out:eval_first_token']:
            label = nodes['full_seq']
            init_w_shape = (1, self._get_batch_size(weight))
            weight = tf.concat([tf.ones(init_w_shape, dtype=tf.float32), weight], 0)
        num_sequences = tf.reduce_sum(seq_weight)
        num_tokens = tf.reduce_sum(weight)

        # likelihood
        c_token_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=label) * weight
        u_token_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=nodes['unigram_logit'], labels=label) * weight

        # combine everything
        loss = c_token_nll + u_token_nll
        loss = tf.reduce_sum(loss) / num_sequences

        # Format output info
        debug_info = {
            'avg.tokens::c_ppl|exp': tf.reduce_sum(c_token_nll) / num_tokens,
            'num.tokens::c_ppl|exp': num_tokens,
            'avg.tokens::u_ppl|exp': tf.reduce_sum(u_token_nll) / num_tokens,
            'num.tokens::u_ppl|exp': num_tokens}
        train_fetch = {'train_loss': loss, 'debug_info': debug_info}
        eval_fetch = {'eval_loss': loss, 'debug_info': debug_info}
        if opt['out:token_nll']:
            eval_fetch['token_nll'] = c_token_nll
        return train_fetch, eval_fetch, {}


class UnigramSeqModelH(UnigramSeqModel):
    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size, reuse_scope, reuse,
            nodes):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        batch_size = self._get_batch_size(lookup)
        input_dim = lookup.shape[-1]
        max_num_tokens = tf.shape(lookup)[0]
        if opt['out:eval_first_token']:
            max_num_tokens += 1
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True) as scope:
            _reuse = reuse or scope is not None
            cell_ = tfg.create_cells(
                input_size=opt['emb:dim'], wrap_state=True, **cell_opt)
            cell_output_, initial_state_, final_state_ = tfg.create_rnn(
                cell_, lookup, seq_len, initial_state, rnn_fn=opt['rnn:fn'],
                batch_size=batch_size)

            u_out = cell_.tiled_init_state(batch_size, max_num_tokens)[-1]
            x = cell_.tiled_init_state(batch_size, 1)[-1]
            if isinstance(u_out, tf.nn.rnn_cell.LSTMStateTuple):
                u_out = u_out.h
                x = x.h
            if opt['cell:out_keep_prob'] < 1.0:
                u_out = tf.nn.dropout(u_out, opt['cell:out_keep_prob'])
            extra_nodes = {'unigram_features': u_out, 'max_num_tokens': max_num_tokens}
            if opt['out:eval_first_token']:
                cell_output_ = tf.concat([x, cell_output_], axis=0)
        return cell_, cell_output_, initial_state_, final_state_, extra_nodes


class BackwardSeqModel(_sqm.SeqModel):

    _FULL_SEQ_ = True

    def _output2state(self, opt, outputs, states):
        self._auto = 0

        def dense(state):
            h = tfdense(
                outputs, state.shape[-1], activation=tf.nn.elu, name=f'h_{self._auto}')
            predict_state = tfdense(
                h, state.shape[-1], activation=tf.nn.tanh, name=f's_{self._auto}')
            self._auto = self._auto + 1
            return predict_state

        return util.nested_map(dense, states)

    def _create_cell(self, opt, get_states=False):
        cell_opt = util.dict_with_key_startswith(opt, 'cell:')
        cell = tfg.create_cells(input_size=opt['emb:dim'], **cell_opt)
        if get_states:
            return tfcell.StateOutputCellWrapper(cell)
        return cell

    def _flatten_state(self, state):
        # XXX: 2 layer lstm cell
        return tf.concat(
            [state[0].c, state[0].h, state[1].c, state[1].h], axis=-1)

    def _build_rnn(
            self, opt, lookup, seq_len, initial_state, batch_size, reuse_scope, reuse,
            nodes):
        concat0 = partial(tf.concat, axis=0)
        new0axis = partial(tf.expand_dims, axis=0)
        full_reverse0 = partial(
            tf.reverse_sequence, seq_lengths=seq_len+1, seq_axis=0, batch_axis=1)
        unroll_rnn = partial(tfg.create_rnn, rnn_fn=opt['rnn:fn'], batch_size=batch_size)
        extra_nodes = {}
        # trace
        with tfg.maybe_scope(reuse_scope[self._RSK_RNN_], reuse=True):
            dec_cell = self._create_cell(opt, get_states=True)
            (g_dec_out, g_h), g_init_h, g_final_h = unroll_rnn(
                dec_cell, lookup, seq_len, initial_state)
            zero_state = dec_cell.zero_state(batch_size, tf.float32)[-1]
            first_output = g_init_h[-1]
            if isinstance(first_output, tf.nn.rnn_cell.LSTMStateTuple):
                first_output = first_output.h
                zero_state = zero_state.h
            full_output = tf.concat(
                (tf.expand_dims(first_output, 0), g_dec_out), 0)
            if opt['out:eval_first_token']:
                g_dec_out = full_output
        # copy
        with tf.variable_scope('encoder'):
            full_seq_lookup = nodes.get('full_lookup', lookup)
            full_seq_lookup = full_reverse0(nodes.get('full_lookup', lookup))
            enc_cell = self._create_cell(opt, get_states=False)
            e_enc_out, __, e_final_h = unroll_rnn(
                enc_cell, full_seq_lookup, seq_len+1, None)
            e_enc_out = full_reverse0(e_enc_out)
            e_enc_out = tfdense(e_enc_out, e_enc_out.shape[-1], activation=tf.nn.elu)
            e_enc_out = tfdense(e_enc_out, e_enc_out.shape[-1], activation=tf.nn.tanh)
        extra_nodes.update(
            unigram_features=zero_state, g_states=full_output, e_states=e_enc_out)
        return dec_cell, g_dec_out, g_init_h, g_final_h, extra_nodes

    # def _build_loss(
    #         self, opt, logit, label, weight, seq_weight, nodes, collect_key,
    #         add_to_collection, inputs=None):
    #     def l2(g_state, e_state):
    #         return tf.reduce_sum(tf.squared_difference(g_state, e_state) / 2, axis=-1)
    #     init_w_shape = (1, self._get_batch_size(weight))
    #     weight = tf.concat([tf.ones(init_w_shape, dtype=tf.float32), weight], 0)
    #     g_states = nodes['g_states']
    #     e_states = nodes['e_states']
    #     # l2_loss = util.nested_map(l2, g_states, e_states)
    #     # l2_loss = tf.add_n(util.flatten(l2_loss)) * weight
    #     l2_loss = l2(g_states, e_states) * weight
    #     # l2_loss = tf.Print(l2_loss, [tf.reduce_mean(l2_loss, 1)])
    #     l2_loss = tf.reduce_sum(l2_loss)
    #     num_sequences = tf.reduce_sum(seq_weight)
    #     num_tokens = tf.reduce_sum(weight)
    #     eval_fetch = {'eval_loss': l2_loss / num_tokens, 'debug_info': {}}
    #     train_fetch = {'train_loss': l2_loss / num_sequences, 'debug_info': {}}
    #     return train_fetch, eval_fetch, {}

