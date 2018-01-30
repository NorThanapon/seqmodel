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
    model_lls = np.load(args.model_path)
    return eval_ngrams, eval_ngram_counts, target_lls, model_lls


def report_error(data):
    under = data[:, -1][data[:, -1] > 0]
    over = data[:, -1][data[:, -1] < 0]
    total_under = under.shape[0]
    total_over = over.shape[0]
    if total_under == 0:
        mean_under = 0.0
    else:
        mean_under = under.mean()
    if total_over == 0:
        mean_over = 0.0
    else:
        mean_over = over.mean()
    print(f'{mean_under}\t{total_under}\t{mean_over}\t{total_over}')


args = _parse_args()
eval_ngrams, eval_ngram_counts, target_lls, model_lls = _load_data(args)
eval_ngram_lens = [len(ngram) for ngram in eval_ngrams]
errors = target_lls - model_lls

eval_data = np.stack([eval_ngram_counts, eval_ngram_lens, errors], axis=-1)

print('by-lengths')
_eval_data = eval_data[eval_data[:, 0] > 1]
for n in range(1, 6):
    n_eval_data = _eval_data[_eval_data[:, 1] == n]
    report_error(n_eval_data)

print('by-counts')
ticks = [0, 2, 5, 10, 50, 100, 500, 2500, 5000, float('inf')]
for i in range(len(ticks) - 1):
    min_count, max_count = ticks[i], ticks[i+1]
    tick_eval_data = eval_data[np.where(
        np.all(np.stack([
            eval_data[:, 0] >= min_count,
            eval_data[:, 0] < max_count], -1), -1))]
    report_error(tick_eval_data)
