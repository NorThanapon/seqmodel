#!/bin/bash

CONFIGDIR="experiments/marginal/ptb-config"
ROOTDIR="curexp"
COMMONOPT="
--gpu --log_level debug --batch_size 64 --seq_len 35
--load_train_opt $CONFIGDIR/train_opt.json
--load_model_opt $CONFIGDIR/model_opt.json
"

# --------------------------------------------------
# Standard training
# --------------------------------------------------
# EXP_DIR="ptb-v"
# rm -r "$ROOTDIR/$EXP_DIR"
# python main_lm.py train data/ptb "$ROOTDIR/$EXP_DIR" $COMMONOPT \
#     --model_class seqmodel.SeqModel

# --------------------------------------------------
# Reset state to zero, and MLE unigram
# --------------------------------------------------
# EXP_DIR="ptb-h0-reset0.1-unigram"
# rm -r "$ROOTDIR/$EXP_DIR"
# python main_lm.py train data/ptb "$ROOTDIR/$EXP_DIR" $COMMONOPT \
#     --cell:reset_state_prob 0.10 \
#     --model_class seqmodel.UnigramSeqModelH

# --------------------------------------------------
# Reset state to training variables, and MLE unigram
# --------------------------------------------------
# EXP_DIR="ptb-h-reset0.1-unigram"
# rm -r "$ROOTDIR/$EXP_DIR"
# python main_lm.py train data/ptb "$ROOTDIR/$EXP_DIR" $COMMONOPT \
#     --cell:reset_state_prob 0.10 \
#     --cell:init_state_trainable \
#     --model_class seqmodel.UnigramSeqModelH
