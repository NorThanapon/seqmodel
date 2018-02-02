import six
from pydoc import locate
from functools import partial
from collections import namedtuple

import numpy as np
import tensorflow as tf

from seqmodel import graph as tfg
from seqmodel.util import nested_map
from seqmodel.dstruct import Pair
from seqmodel.dstruct import OutputStateTuple

tfdense = tf.layers.dense


ResetCellOutput = namedtuple('ResetCellOutput', 'output reset')


class InitStateCellWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(
            self, cell, state_reset_prob=0.0, trainable=False,
            dtype=tf.float32, actvn=tf.nn.tanh, output_reset=False):
        self._cell = cell
        self._init_vars = self._create_init_vars(trainable, dtype, actvn)
        self._dtype = dtype
        self._reset_prob = state_reset_prob
        self._actvn = actvn
        self._output_reset = output_reset

    @property
    def output_size(self):
        if self._output_reset:
            return ResetCellOutput(self._cell.output_size, 1)
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def init_state(self):
        return self._init_vars

    def _create_init_vars(self, trainable, dtype, actvn=None):
        self._i = 0
        with tf.variable_scope('init_state'):
            def create_init_var(size):
                var = tf.get_variable(
                    f'init_{self._i}', shape=(size, ), dtype=dtype,
                    initializer=tf.zeros_initializer(), trainable=trainable)
                if actvn is not None:
                    var = actvn(var)
                self._i = self._i + 1
                return var
            if (isinstance(self.state_size[0], tf.nn.rnn_cell.LSTMStateTuple) and
               trainable):
                return self._create_lstm_init_vars(trainable, dtype)
            return nested_map(create_init_var, self.state_size)

    def _create_lstm_init_vars(self, trainable, dtype):
        num_layers = len(self.state_size)
        states = []
        for i in range(num_layers):
            state_size = self.state_size[i]
            assert isinstance(state_size, tf.nn.rnn_cell.LSTMStateTuple), \
                '`state_size` is not LSTMStateTuple'
            c = tf.get_variable(
                f'init_{i}_c', shape=(state_size.c, ), dtype=dtype,
                trainable=trainable)
            h = tf.get_variable(
                f'init_{i}_h', shape=(state_size.c, ), dtype=dtype,
                trainable=trainable)
            c = tf.clip_by_value(c, -1.0, 1.0)
            h = tf.tanh(h)
            # h = tf.Print(h, [tf.reduce_mean(h)])
            states.append(tf.nn.rnn_cell.LSTMStateTuple(c, h))
        return tuple(states)

    def _get_reset(self, inputs):
        # TODO: better way to figure out the batch size
        batch_size = tf.shape(inputs)[0]
        rand = tf.random_uniform((batch_size, ))
        r = tf.cast(tf.less(rand, self._reset_prob), tf.float32)
        r = r[:, tf.newaxis]
        return r, batch_size

    def _get_zero_reset(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.zeros((batch_size, 1), dtype=tf.float32)

    def __call__(self, inputs, state, scope=None):
        r = None
        if self._reset_prob > 0.0:
            r, batch_size = self._get_reset(inputs)
            state = self.select_state(state, r, batch_size)
        cell_output, new_state = self._cell(inputs, state)
        if self._output_reset:
            if self._reset_prob <= 0.0:
                r = self._get_zero_reset(inputs)
            return ResetCellOutput(cell_output, r), new_state
        else:
            return cell_output, new_state

    def select_state(self, state, r, batch_size):
        def _select(cur_state, init_var):
            return r * (init_var[tf.newaxis, :] - cur_state) + cur_state
        return nested_map(_select, state, self._init_vars)

    def tiled_init_state(self, batch_size, seq_len=None):
        def _tile(var):
            if seq_len is not None:
                return tf.tile(var[tf.newaxis, tf.newaxis, :], (seq_len, batch_size, 1))
            return tf.tile(var[tf.newaxis, :], (batch_size, 1))
        return nested_map(_tile, self._init_vars)

    def zero_state(self, batch_size, dtype):
        assert dtype == self._dtype, \
            'dtype must be the same as dtype during the cell construction'
        return self.tiled_init_state(batch_size)


AEStateCellInput = namedtuple('AEStateCellInput', 'inputs qstate')


class NormalInitStateCellWrapper(InitStateCellWrapper):

    _SCALE_INIT_ = 0.5  # softplus(0.5) = 0.974, closed enough to 1.0

    def __init__(
            self, cell, state_reset_prob=0.0, trainable=False,
            dtype=tf.float32, actvn=None, output_reset=False,
            mean_actvn=tf.nn.tanh, scale_actvn=tf.nn.softplus):
        self._mean_actvn = mean_actvn
        self._scale_actvn = scale_actvn
        super().__init__(cell, state_reset_prob, trainable, dtype, actvn, output_reset)

    def __call__(self, inputs, state, scope=None):
        if isinstance(inputs, AEStateCellInput):
            inputs, q_state = inputs
        else:
            q_state = None
        if self._reset_prob > 0.0:
            r, batch_size = self._get_reset(inputs)
            state = self.select_state(
                state, r, batch_size, injected_states=q_state)
        cell_output, new_state = self._cell(inputs, state)
        if self._output_reset:
            if self._reset_prob <= 0.0:
                r = self._get_zero_reset(inputs)
            return ResetCellOutput(cell_output, r), new_state
        else:
            return cell_output, new_state

    def _create_init_vars(self, trainable, dtype, actvn=None):
        self._i = 0
        with tf.variable_scope('init_state'):
            def create_init_var(size):
                # mean = tf.get_variable(
                #     f'init_mean_{self._i}', shape=(size, ), dtype=dtype,
                #     initializer=tf.zeros_initializer(), trainable=trainable)
                mean = self._mean_actvn(tf.get_variable(
                    f'init_mean_{self._i}', shape=(size, ), dtype=dtype,
                    initializer=tf.zeros_initializer(), trainable=trainable))

                scale = self._scale_actvn(tf.get_variable(
                    f'init_scale_{self._i}', shape=(size, ), dtype=dtype,
                    initializer=tf.constant_initializer(value=self._SCALE_INIT_),
                    trainable=trainable))
                self._i = self._i + 1
                return Pair(mean, scale)
            return nested_map(create_init_var, self.state_size)

    def tiled_init_state(self, batch_size, seq_len=None):
        def _tile(var):
            mean, scale = var
            if seq_len is not None:
                mean = tf.tile(mean[tf.newaxis, tf.newaxis, :], (seq_len, batch_size, 1))
                scale = scale[tf.newaxis, tf.newaxis, :]
            else:
                mean = tf.tile(mean[tf.newaxis, :], (batch_size, 1))
                scale = scale[tf.newaxis, :]
            sample = mean + scale * tf.random_normal(tf.shape(mean))
            if self._actvn is not None:
                sample = self._actvn(sample)
            return sample
        return nested_map(_tile, self._init_vars)

    def select_state(self, state, r, batch_size, injected_states=None):
        need_tile = injected_states is None

        def _select(cur_state, var):
            mean, scale = var
            if need_tile:
                mean = tf.tile(mean[tf.newaxis, :], (batch_size, 1))
                scale = scale[tf.newaxis, :]
            init_state = mean + scale * tf.random_normal(tf.shape(mean))
            if self._actvn is not None:
                init_state = self._actvn(init_state)
            return r * (init_state - cur_state) + cur_state

        if injected_states is None:
            injected_states = self._init_vars
        return nested_map(_select, state, injected_states)

    def zero_state(self, batch_size, dtype):
        assert dtype == self._dtype, \
            'dtype must be the same as dtype during the cell construction'
        return self.tiled_init_state(batch_size)


class StateOutputCellWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell):
        self._cell = cell

    @property
    def output_size(self):
        return OutputStateTuple(self._cell.output_size, self._cell.state_size)

    @property
    def state_size(self):
        return self._cell.state_size

    def __call__(self, inputs, state, scope=None):
        output, new_state = self._cell(inputs, state, scope=scope)
        return OutputStateTuple(output, new_state), new_state


class AttendedInputCellWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, each_input_dim=None, attention_fn=None):
        self._cell = cell
        self._each_input_dim = each_input_dim
        if attention_fn is None:
            attention_fn = partial(
                tfg.attend_dot,
                time_major=False, out_q_major=False,
                gumbel_select=True, gumbel_temperature=4.0)
        self._attention_fn = attention_fn

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return OutputStateTuple(self.output_size, self._cell.state_size)

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(f'{type(self).__name__}_ZeroState', values=[batch_size]):
            return OutputStateTuple(
                tf.zeros((batch_size, self.output_size), dtype=dtype),
                self._cell.zero_state(batch_size, dtype))

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            prev_state, prev_output = state
            if self._each_input_dim is not None:
                input_shape = (tf.shape(inputs)[0], -1, self._each_input_dim)  # (b, n, ?)
                inputs = tf.reshape(inputs, input_shape)
            attn_inputs, scores = self._attention_fn(
                prev_output, inputs, inputs)  # (b, ?) and (b, n)
            output, state = self._cell(attn_inputs, prev_state)
            state = OutputStateTuple(output, state)
            return output, state


GaussianState = namedtuple('GaussianState', 'mean scale sample')


class GaussianCellWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(
            self, cell, num_hidden_layers=1, hidden_actvn=tf.nn.elu,
            mean_actvn=None, scale_actvn=tf.nn.softplus, sample_actvn=None):
        self._cell = cell
        self._num_layers = num_hidden_layers
        self._hactvn = hidden_actvn
        if isinstance(self._hactvn, six.string_types):
            self._hactvn = locate(self._hactvn)
        self._mactvn = mean_actvn
        if isinstance(self._mactvn, six.string_types):
            self._mactvn = locate(self._mactvn)
        self._sactvn = scale_actvn
        if isinstance(self._sactvn, six.string_types):
            self._sactvn = locate(self._sactvn)
        self._zactvn = sample_actvn
        if isinstance(self._zactvn, six.string_types):
            self._zactvn = locate(self._zactvn)

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        cell_state_size = self._cell.state_size
        return GaussianState(*tuple((cell_state_size, ) * 3))

    def __call__(self, inputs, state, scope=None):
        if isinstance(state, GaussianState):
            __, __, prev_sample = state
        else:
            prev_sample = state
        new_output, new_state = self._cell(inputs, prev_sample)
        # XXX: make it work with LSTMCell
        with tf.variable_scope(scope or 'gaussian_wrapper'):
            gauss_input = new_state
            for i in range(self._num_layers):
                h = tfdense(
                    gauss_input, gauss_input.shape[-1], activation=self._hactvn,
                    name=f'hidden_{i}')
                gauss_input + h
            mean, scale = tf.split(
                tfdense(gauss_input, gauss_input.shape[-1] * 2, name='gauss'), 2, axis=-1)
            if self._sactvn is not None:
                scale = self._sactvn(scale)
            if self._mactvn is not None:
                mean = self._mactvn(mean)
            max_scale = tf.constant(gauss_input.shape[-1].value, dtype=tf.float32)
            scale = tf.minimum(max_scale, scale)
            noise = scale * tf.random_normal(tf.shape(mean))
            new_state = mean + noise
            if self._zactvn is not None:
                new_state = self._zactvn(new_state)
            new_output = new_state
            return new_output, GaussianState(mean, scale, new_state)
