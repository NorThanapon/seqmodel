#!/bin/bash

EXP_DIR='curexp/ptb-temp'
rm -r $EXP_DIR
python main_lm.py train data/ptb $EXP_DIR \
--gpu --log_level debug --batch_size 64 --seq_len 35 \
--load_train_opt "curexp/ptb-config/train_opt.json" \
--load_model_opt "curexp/ptb-config/model_opt.json" \
--load_checkpoint "curexp/ptb-v/checkpoint/best" \
--relax_ckp_restore \
--random_seq_len --random_seq_len_min 3 --random_seq_len_max 13 \
--model_class seqmodel.BackwardSeqModel \
--train:grad_vars_contain "encoder"


# CONFIGDIR="experiments/marginal/config"
# ROOTDIR="curexp"
# COMMONOPT="
# --gpu --log_level debug --batch_size 64 --seq_len 35
# --load_train_opt $CONFIGDIR/train_opt-med.json
# --load_model_opt $CONFIGDIR/model_opt-med.json
# "

# # --------------------------------------------------
# # Standard training
# # --------------------------------------------------
# EXP_DIR="wt2-v-med-backward"
# rm -r "$ROOTDIR/$EXP_DIR"
# python main_lm.py train data/wikitext-2 "$ROOTDIR/$EXP_DIR" $COMMONOPT \
#     --model_class seqmodel.BackwardSeqModel \
#     --train:grad_vars_contain "encoder" \
#     --load_checkpoint "curexp/wt2-h0-reset0.05-unigram-med/checkpoint/best" \
# 	--relax_ckp_restore \
# 	--random_seq_len --random_seq_len_min 7 --random_seq_len_max 13
