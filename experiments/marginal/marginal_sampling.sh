#!/bin/bash

# PTB

# ipython script/compute_marginals.py -- \
# curexp/ptb-v-backward/ curexp/ptb-v-backward/marginals/sample.5.count-filter-10 \
# --method trace \
# --output_filename sample.5.count-filter-10.rand_trace_100.lls \
# --gpu --batch_size 32 --num_samples 100 --vocab_path data/ptb/vocab.txt \
# --num_trace_splits 1 --mini_sample_size 10 --trace_random --repeat 30

# ipython script/compute_marginals.py -- \
# curexp/ptb-v-backward/ curexp/ptb-v-backward/marginals/sample.5.count-filter-10 \
# --method trace \
# --output_filename sample.5.count-filter-10.iw_trace_100.lls \
# --gpu --batch_size 32 --num_samples 100 --vocab_path data/ptb/vocab.txt \
# --num_trace_splits 1 --mini_sample_size 10 --repeat 30 --gpu_trace

# WT2

ipython script/compute_marginals.py -- \
curexp/wt2-v-med-backward/ curexp/wt2-v-med-backward/marginals/sample.5.count-filter-10 \
--method trace \
--output_filename sample.5.count-filter-10.rand_trace_100.lls \
--gpu --batch_size 32 --num_samples 100 --vocab_path data/wikitext-2/vocab.txt \
--num_trace_splits 4 --mini_sample_size 2 --trace_random --repeat 30

ipython script/compute_marginals.py -- \
curexp/wt2-v-med-backward/ curexp/wt2-v-med-backward/marginals/sample.5.count-filter-10 \
--method trace \
--output_filename sample.5.count-filter-10.iw_trace_100.lls \
--gpu --batch_size 32 --num_samples 100 --vocab_path data/wikitext-2/vocab.txt \
--num_trace_splits 4 --mini_sample_size 2 --repeat 30 --gpu_trace

# ipython script/compute_marginals.py -- \
# curexp/wt2-v-med-backward/ curexp/wt2-v-med-backward/marginals/sample.5.count-filter \
# --method trace \
# --output_filename sample.5.count-filter.iw_trace_20.lls \
# --gpu --batch_size 32 --num_samples 20 --vocab_path data/wikitext-2/vocab.txt \
# --num_trace_splits 4 --gpu_trace --mini_sample_size 2
