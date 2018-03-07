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
        prog='compute_marginal_error',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('eval_path', type=str, help='a file listing n-grams to compute')
    parser.add_argument('target_path', type=str, help='')
    parser.add_argument('model_path', type=str, help='')
    parser.add_argument('--num_results', type=int, default=30)
    args = parser.parse_args()
    return args


def _load_data(args):
    eval_ngrams = []
    eval_ngram_counts = []
    batch_lines = []
    with open(args.eval_path) as lines:
        for line in lines:
            line, count = line.strip().split('\t')
            ngram = tuple(line.split(' '))
            eval_ngrams.append(ngram)
            eval_ngram_counts.append(int(count))
    target_lls = np.load(args.target_path)
    model_lls = []
    for i in range(args.num_results):
        model_lls.append(np.load(args.model_path.replace('-n-', str(i))))
    return eval_ngrams, eval_ngram_counts, target_lls, model_lls


def report_error(args, data):
    k = args.num_results
    mean_error = data[:, -k:].mean()
    mean_var = data[:, -k:].var(axis=-1).mean()
    print(f'{mean_error}\t{mean_var}')


args = _parse_args()
eval_ngrams, eval_ngram_counts, target_lls, model_lls = _load_data(args)
model_lls = np.stack(model_lls, axis=-1)
eval_ngram_lens = [len(ngram) for ngram in eval_ngrams]
errors = np.abs(target_lls[:, np.newaxis] - model_lls)
mean_errors = np.mean(errors, axis=-1)
var_errors = np.var(errors, axis=-1)

# eval_data = np.stack(
#     [eval_ngram_counts, eval_ngram_lens, mean_errors, var_errors], axis=-1)

eval_data = np.stack([eval_ngram_counts, eval_ngram_lens], axis=-1)
eval_data = np.concatenate((eval_data, errors), axis=-1)

print('by-lengths')
_eval_data = eval_data[eval_data[:, 0] >= 10]
for n in range(1, 6):
    n_eval_data = _eval_data[_eval_data[:, 1] == n]
    report_error(args, n_eval_data)

print('by-counts')
ticks = [10, 20, 50, 100, 500, float('inf')]
for i in range(len(ticks) - 1):
    min_count, max_count = ticks[i], ticks[i+1]
    tick_eval_data = eval_data[np.where(
        np.all(np.stack([
            eval_data[:, 0] >= min_count,
            eval_data[:, 0] < max_count], -1), -1))]
    report_error(args, tick_eval_data)
