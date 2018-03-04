import os
import sys
# import time
import json
import pickle
import argparse
from pydoc import locate
from functools import partial
from collections import ChainMap
from itertools import chain

import numpy as np
import h5py
import tensorflow as tf

sys.path.insert(0, '../')
import seqmodel as sq  # noqa


def _parse_args():
    parser = argparse.ArgumentParser(
        prog='compute_marginals', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path', type=str, help='')
    parser.add_argument('eval_path', type=str, help='a file listing n-grams to compute')
    parser.add_argument('--output_filename', type=str, default=None, help='')
    parser.add_argument('--gpu', action='store_true', help='')
    parser.add_argument('--num_threads', type=int, default=8, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--vocab_path', type=str, default='data/ptb/vocab.txt', help='')
    parser.add_argument('--method', type=str, default='init', help='')
    parser.add_argument('--trace_name', type=str, default='train.txt')
    parser.add_argument('--trace_random', action='store_true')
    parser.add_argument('--gpu_trace', action='store_true')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--mini_sample_size', type=int, default=2)
    parser.add_argument('--num_trace_splits', type=int, default=1)
    args = parser.parse_args()
    return args


def _load_data(args):
    vocab = sq.Vocabulary.from_vocab_file(args.vocab_path)
    eval_ngrams = []
    batch_lines = []
    with open(args.eval_path) as lines:
        for line in lines:
            line = line.strip().split('\t')[0]
            ngram = tuple(line.split(' '))
            eval_ngrams.append(ngram)
            if len(ngram) > 1:
                batch_lines.append((ngram, ))
    data = sq.read_seq_data(
        batch_lines, vocab, vocab, keep_sentence=True,
        data_include_eos=True, add_sos=False)
    batches = partial(
        sq.seq_batch_iter, *data, batch_size=args.batch_size,
        shuffle=False, keep_sentence=True)
    return vocab, eval_ngrams, batches


def _load_count_file(args):
    exp_path = partial(os.path.join, args.model_path, 'marginals')
    ngram_counts = sq.ngram_stat.read_ngram_count_file(
        exp_path('ngrams.count'), min_order=-1, max_order=-1)
    if os.path.exists(exp_path('total_tokens.txt')):
        with open(exp_path('total_tokens.txt')) as line:
            total_tokens = int(line.readline())
    elif os.path.exists(exp_path('unigrams.count')):
        total_tokens = 0
        with open(exp_path('unigrams.count')) as lines:
            for line in lines:
                total_tokens += int(line.strip().split('\t')[-1])
    else:
        raise FileNotFoundError(
            f'{exp_path} does not contain `total_tokens.txt` or `unigrams.count`.')
    return ngram_counts, total_tokens


def _load_model(args, vocab):
    exp_path = partial(os.path.join, args.model_path)
    with open(exp_path('basic_opt.json')) as fp:
        basic_opt = json.load(fp)
        model_class = basic_opt['model_class']
        if model_class == '':
            model_class = 'seqmodel.SeqModel'
        MODEL_CLASS = locate(model_class)
    with open(exp_path('model_opt.json')) as ifp:
        model_opt = json.load(ifp)
    model_vocab_opt = MODEL_CLASS.get_vocab_opt(*(v.vocab_size for v in [vocab, vocab]))
    model_opt = ChainMap(
        {'out:token_nll': True, 'out:eval_first_token': True}, model_vocab_opt, model_opt)
    model = MODEL_CLASS()
    nodes = model.build_graph(model_opt, no_dropout=True)
    sess_config = sq.get_tfsession_config(args.gpu, args.num_threads)
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    filter_var_map = sq.filter_tfvars_in_checkpoint(
        tf.global_variables(), exp_path('checkpoint/best'))
    # saver = tf.train.Saver(filter_var_map)
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, exp_path('checkpoint/best'))
    return model, nodes, sess


def _load_trace_states(args):
    trace_name = args.trace_name
    h5py_filename = os.path.join(
        args.model_path, 'marginals', f'{trace_name}-trace-clean.h5')
    if os.path.exists(h5py_filename):
        print('...h5py trace data...')
        with h5py.File(h5py_filename, 'r') as f:
            clean_trace_states = f['train-trace-clean'][:, :]
    else:
        print('...npy trace data...')
        trace_states = np.load(
            os.path.join(args.model_path, 'marginals', f'{trace_name}-trace.npy'))
        # TODO: preserve batch
        trace_states = np.reshape(trace_states, [-1, trace_states.shape[-1]])
        clean_trace_states = trace_states[~np.all(trace_states == 0, axis=-1)]
        del trace_states
    return clean_trace_states


def compute_count_ll(eval_ngrams, ngram_counts, total_tokens):
    eval_lls = []
    log_total_count = np.log(total_tokens)
    for ngram in eval_ngrams:
        count = ngram_counts[ngram]
        if count == 0:
            print(ngram)
            raise ValueError('zero count!')
        eval_lls.append(np.log(count) - log_total_count)
    return eval_lls


def iw_trace_ll(sess, eval_fn, batch, trace_obj, unigram=False):
    num_samples = trace_obj['num_samples']
    __, extra = eval_fn(batch.features, batch.labels, extra_fetch=['e_states'])
    first_state = extra[0][0]
    if unigram:
        first_state = extra[0][-1]
    token_lls = []
    log_scores = []
    mini_sample_size = min(trace_obj['mini_sample_size'], num_samples)
    all_log_scores, __ = sess.run(
            [trace_obj['tf_trace_log_scores'], trace_obj['tf_update_cache']],
            {trace_obj['tf_trace_q']: first_state})
    for j in range(num_samples//mini_sample_size):
        choices = sess.run(
            trace_obj['tf_cache_trace_choices'],
            {trace_obj['tf_trace_num']: mini_sample_size})
        for i in range(mini_sample_size):
            state = make_state(trace_obj['trace'][choices[:, i]])
            result, __ = eval_fn(batch.features, batch.labels, state=state)
            token_lls.append(-result['token_nll'])
        _log_scores = []
        for i in range(len(trace_obj['batch_size'])):
            _log_scores.append(all_log_scores[i][choices[i]])
        log_scores.append(np.stack(_log_scores))
    weights = np.log(1 / len(trace_obj['trace'])) - np.concatenate(log_scores, -1)
    token_lls = np.stack(token_lls, axis=-1)
    # XXX: first token is weighted
    token_lls[0, :, :] = token_lls[0, :, :] + weights
    avg_tokens_ll = sq.log_sumexp(token_lls, axis=-1) - np.log(num_samples)
    if unigram:
        return avg_tokens_ll[0]
    return avg_tokens_ll


def random_trace_ll(sess, eval_fn, batch, trace_obj, unigram=False):
    num_samples = trace_obj['num_samples']
    token_lls = []
    for j in range(num_samples):
        choices = get_random_state_ids(len(batch.features.seq_len))
        state = make_state(trace_obj['trace'][choices])
        result, __ = eval_fn(batch.features, batch.labels, state=state)
        token_lls.append(-result['token_nll'])
    token_lls = np.stack(token_lls, axis=-1)
    avg_tokens_ll = sq.log_sumexp(token_lls, axis=-1) - np.log(num_samples)
    if unigram:
        return avg_tokens_ll[0]
    return avg_tokens_ll


def compute_unigram_ll(
        vocab, sess, nodes, method, batch_size=32, trace_obj=None):
    if method == 'init':
        feed_dict = {nodes['temperature']: 1.0, nodes['batch_size']: 1}
        if 'max_num_tokens' in nodes:
            feed_dict[nodes['max_num_tokens']] = 1
        unigram_ll = sess.run(nodes['log_u_dist'], feed_dict)
        unigram_ll = np.squeeze(unigram_ll, axis=0)
    elif method == 'trace':
        batch_words = []
        word_set = vocab.word_set()
        unigram_ll = np.zeros((len(word_set), ), np.float32)
        iter_word_set = word_set
        if len(word_set) % batch_size != 0:
            pads = ['</s>'] * (batch_size - len(word_set) % batch_size)
            iter_word_set = chain(word_set, pads)
        for word in iter_word_set:
            batch_words.append((word, word))
            if len(batch_words) == batch_size:
                batch = make_batch(vocab, batch_words)
                batch_lls = trace_obj['trace_ll_fn'](
                    sess, eval_fn, batch, trace_obj, unigram=True)
                for bw, ll in zip(batch_words, batch_lls):
                    unigram_ll[vocab[bw[0]]] = ll
                del batch_words[:]
        if len(batch_words) > 0:
            raise ValueError('vocab size is not divisible by batch size')
            # batch = make_batch(vocab, batch_words)
            # batch_lls = trace_obj['trace_ll_fn'](
            #     sess, eval_fn, batch, trace_obj, unigram=True)
            # for bw, ll in zip(batch_words, batch_lls):
            #     unigram_ll[vocab[bw[0]]] = ll
    return unigram_ll.squeeze()


def compute_ll(eval_fn, batch, method, sess, trace_obj=None):
    inputs, labels = batch.features, batch.labels
    max_steps = batch.features.inputs.shape[0] + 1
    batch_size = batch.features.inputs.shape[-1]
    if method == 'init':
        result, __ = eval_fn(inputs, labels)
        token_ll = -result['token_nll']
    elif method == 'trace':
        token_ll = trace_obj['trace_ll_fn'](
            sess, eval_fn, batch, trace_obj, unigram=False)
    return token_ll


def make_state(vector):
    # XXX: 2-layer LSTM cell
    split_vector = np.split(vector, 4, axis=-1)
    state = tuple((
        tf.nn.rnn_cell.LSTMStateTuple(split_vector[0], split_vector[1]),
        tf.nn.rnn_cell.LSTMStateTuple(split_vector[2], split_vector[3])))
    return state


def make_batch(vocab, ngrams):
    ids = np.array(vocab.w2i(ngrams))
    x, y = ids[:, :-1], ids[:, 1:]
    x_arr, x_len = sq.hstack_list(x)
    y_arr, y_len = sq.hstack_list(y)
    seq_weight = np.where(y_len > 0, 1, 0).astype(np.float32)
    token_weight, num_tokens = sq.masked_full_like(
        y_arr, 1, num_non_padding=y_len)
    features = sq.SeqFeatureTuple(x_arr, x_len)
    labels = sq.SeqLabelTuple(y_arr, token_weight, seq_weight)
    batch = sq.BatchTuple(features, labels, num_tokens, False)
    return batch


args = _parse_args()
method = args.method
print('Loading data...')
vocab, eval_ngrams, batches = _load_data(args)
trace_obj = None
if method == 'trace':
    trace = _load_trace_states(args)
    state_size = trace.shape[-1] // 4  # XXX: 2-layer LSTM cell
    trace_key = trace[:, -state_size:].T
    # trace_key = trace.T
    trace_obj = {
        'trace_ll_fn': iw_trace_ll, 'trace': trace, 'trace_key': trace_key,
        'num_samples': args.num_samples, 'batch_size': args.batch_size,
        'mini_sample_size': args.mini_sample_size
    }
    tf_trace_assigns = []
    if args.gpu_trace:
        print('initializing TF trace computation...')
        tf_trace_q = tf.placeholder(dtype=tf.float32, shape=(None, state_size))
        tf_trace_num = tf.placeholder(dtype=tf.int32, shape=None)
        if args.num_trace_splits > 1:
            chunks = np.array_split(trace_key, args.num_trace_splits, axis=-1)
            tf_scores = []
            for i, chunk in enumerate(chunks):
                # c_tf_trace_key = tf.constant(chunk, dtype=tf.float32)
                c_tf_trace_key = tf.get_variable(
                    f'trace_{i}', shape=chunk.shape, trainable=False, dtype=tf.float32)
                tf_scores.append(tf.matmul(tf_trace_q, c_tf_trace_key))
                c_tf_trace_ph = tf.placeholder(tf.float32, shape=chunk.shape)
                c_tf_assign = tf.assign(c_tf_trace_key, c_tf_trace_ph)
                tf_trace_assigns.append((c_tf_assign, c_tf_trace_ph, chunk))
            tf_trace_scores = tf.concat(tf_scores, axis=-1)
        else:
            tf_trace_key = tf.constant(trace_key, dtype=tf.float32)
            tf_trace_scores = tf.matmul(tf_trace_q, tf_trace_key)
        tf_trace_log_scores = tf.nn.log_softmax(tf_trace_scores)
        tf_trace_choices = tf.multinomial(
            tf_trace_scores, tf_trace_num, output_dtype=tf.int32)
        tf_cached_scores = tf.get_variable(
            'cached_trace_scores', shape=(args.batch_size, trace_key.shape[-1]),
            dtype=tf.float32, trainable=False)
        tf_update_cache = tf.assign(tf_cached_scores, tf_trace_scores)
        tf_cache_trace_choices = tf.multinomial(
            tf_cached_scores, tf_trace_num, output_dtype=tf.int32)
        trace_obj.update({
            'tf_trace_q': tf_trace_q,
            'tf_trace_scores': tf_trace_scores,
            'tf_trace_log_scores': tf_trace_log_scores,
            'tf_cached_scores': tf_cached_scores, 'tf_update_cache': tf_update_cache,
            'tf_cache_trace_choices': tf_cache_trace_choices,
            'tf_trace_choices': tf_trace_choices, 'tf_trace_num': tf_trace_num,
            })
    if args.trace_random:
        trace_obj['trace_ll_fn'] = random_trace_ll
    get_random_state_ids = partial(np.random.choice, np.arange(len(trace)))
print('Computing marginals...')
if method == 'count':
    ngram_counts, total_tokens = _load_count_file(args)
    eval_lls = compute_count_ll(eval_ngrams, ngram_counts, total_tokens)
elif method in ('init', 'trace'):
    model, nodes, sess = _load_model(args, vocab)
    eval_fn = partial(model.evaluate, sess)
    if len(tf_trace_assigns) > 0:
        print('assigning trace data to TF...')
        for assign_op, ph, chunk in tf_trace_assigns:
            sess.run(assign_op, feed_dict={ph: chunk})
        del tf_trace_assigns
    print('... unigrams ...')
    unigram_lls = compute_unigram_ll(
        vocab, sess, nodes, method, batch_size=args.batch_size, trace_obj=trace_obj)
    print('... n-grams ...')
    ngram_lls = {}
    _count_progress = 0
    for batch in batches():
        token_ll = compute_ll(eval_fn, batch, method, sess, trace_obj=trace_obj)
        token_ll = token_ll[:, batch.features.seq_len != 0]
        sum_ll = np.sum(token_ll, axis=0)
        ngram_idx = np.concatenate([batch.features.inputs[0:1, :], batch.labels.label])
        for i in range(len(sum_ll)):
            ngram = vocab.i2w(ngram_idx[:batch.features.seq_len[i] + 1, i])
            ngram_lls[tuple(ngram)] = sum_ll[i]
            _count_progress += 1
        if _count_progress % 100 == 0:
            print(_count_progress)
    eval_lls = []
    for ngram in eval_ngrams:
        if len(ngram) == 1:
            eval_lls.append(unigram_lls[vocab[ngram[0]]])
        else:
            eval_lls.append(ngram_lls[ngram])
else:
    raise ValueError('method is not valid')

print('Writing output file...')
eval_lls = np.array(eval_lls)
out_directory = os.path.join(args.model_path, 'marginals')
sq.ensure_dir(out_directory)
out_filename = args.output_filename
if out_filename is None:
    basename = os.path.basename(args.eval_path)
    out_filename = f'{basename}-{time.time()}-lls'
np.save(os.path.join(out_directory, out_filename), eval_lls)
