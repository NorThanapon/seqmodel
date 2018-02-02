import os
import sys
import time
import json
import pickle
import argparse
from pydoc import locate
from functools import partial
from collections import ChainMap

import numpy as np
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


def compute_unigram_ll(sess, nodes, method):
    unigram_nlls = []
    if method == 'init':
        feed_dict = {nodes['temperature']: 1.0, nodes['batch_size']: 1}
        if 'max_num_tokens' in nodes:
            feed_dict[nodes['max_num_tokens']] = 1
        unigram_nll = sess.run(
            nodes['log_u_dist'], feed_dict)
        unigram_nll = np.squeeze(unigram_nll, axis=0)
        unigram_nlls.append(unigram_nll)
    nll_count = len(unigram_nlls)
    unigram_nlls = np.stack(unigram_nlls, axis=-1)
    unigram_nll = sq.log_sumexp(unigram_nlls, axis=-1) - np.log(nll_count)
    return unigram_nll.squeeze()


def compute_ll(eval_fn, batch, method):
    inputs, labels = batch.features, batch.labels
    max_steps = batch.features.inputs.shape[0] + 1
    batch_size = batch.features.inputs.shape[-1]
    token_nlls = []
    if method == 'init':
        result, __ = eval_fn(inputs, labels)
        token_nlls.append(result['token_nll'])
    nll_count = len(token_nlls)
    token_nlls = np.stack(token_nlls, -1)
    token_nll = sq.log_sumexp(token_nlls, axis=-1) - np.log(nll_count)
    return token_nll


def make_batch(vocab, ngram, batch_size):
    ids = vocab.w2i(ngram)
    x, y = ids[:-1], ids[1:]
    x = [x for __ in range(batch_size)]
    y = [y for __ in range(batch_size)]
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
print('Computing marginals...')
if method == 'count':
    ngram_counts, total_tokens = _load_count_file(args)
    eval_lls = compute_count_ll(eval_ngrams, ngram_counts, total_tokens)
else:
    model, nodes, sess = _load_model(args, vocab)
    eval_fn = partial(model.evaluate, sess)
    unigram_lls = compute_unigram_ll(sess, nodes, method)
    ngram_lls = {}
    for batch in batches():
        token_nll = compute_ll(eval_fn, batch, method)
        token_nll = token_nll[:, batch.features.seq_len != 0]
        sum_ll = -1.0 * np.sum(token_nll, axis=0)
        ngram_idx = np.concatenate([batch.features.inputs[0:1, :], batch.labels.label])
        for i in range(len(sum_ll)):
            ngram = vocab.i2w(ngram_idx[:batch.features.seq_len[i] + 1, i])
            ngram_lls[tuple(ngram)] = sum_ll[i]
    eval_lls = []
    for ngram in eval_ngrams:
        if len(ngram) == 1:
            eval_lls.append(unigram_lls[vocab[ngram[0]]])
        else:
            eval_lls.append(ngram_lls[ngram])

print('Writing output file...')
eval_lls = np.array(eval_lls)
out_directory = os.path.join(args.model_path, 'marginals')
sq.ensure_dir(out_directory)
out_filename = args.output_filename
if out_filename is None:
    basename = os.path.basename(args.eval_path)
    out_filename = f'{basename}-{time.time()}-lls'
np.save(os.path.join(out_directory, out_filename), eval_lls)
