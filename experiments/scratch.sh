#!/bin/bash

EXP_DIR='tmp1'
rm -r $EXP_DIR
# python main_lm.py train data/ptb $EXP_DIR \
# --gpu --log_level debug --batch_size 64 --seq_len 35 \
# --emb:dim 200 --cell:num_units 200 --cell:num_layers 2 \
# --cell:cell_class tensorflow.contrib.rnn.GRUBlockCellV2 \
# --lr:decay_every 1 --lr:decay_factor 0.98 --lr:start_decay_at 1 \
# --train:max_epoch 20 --train:init_lr 0.001 \
# --model_class seqmodel.VAESeqModel \
# --share:input_emb_logit

python main_lm.py train data/ptb $EXP_DIR \
--gpu --log_level debug --batch_size 64 --seq_len 35 \
--emb:dim 200 --cell:num_units 200 --cell:num_layers 2 \
--cell:cell_class tensorflow.contrib.rnn.GRUBlockCellV2 \
--cell:reset_state_prob 0.10 \
--share:input_emb_logit \
--train:init_lr 0.003 \
--train:max_epoch 20 \
--lr:decay_every 1 --lr:decay_factor 0.8 --lr:start_decay_at 1 \
--model_class seqmodel.UnigramSeqModelH
# --cell:init_state_trainable \
# --load_train_opt "curexp/ptb-config/train_opt.json" \
