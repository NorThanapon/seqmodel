#!/bin/bash

METHODS="init"

main () {
    EXP_DIR=$1
    MODEL_CLASS=$2

    source experiments/marginal/_decode_trace.sh "data/ptb" $EXP_DIR $MODEL_CLASS

    ipython script/compute_marginals.py --  --gpu --batch_size 256 \
        $EXP_DIR "$EXP_DIR/marginals/sample.5.count-filter" \
        --method "count" --output_filename "sample.5.count-filter.count.lls"

    for METHOD in $METHODS ; do
        # sample data
        ipython script/compute_marginals.py --  --gpu --batch_size 256 \
            $EXP_DIR "$EXP_DIR/marginals/sample.5.count-filter" \
            --method $METHOD --output_filename "sample.5.count-filter.$METHOD.lls"
        ipython script/compute_marginal_error.py -- \
            "$EXP_DIR/marginals/sample.5.count-filter" \
            "$EXP_DIR/marginals/sample.5.count-filter.count.lls.npy" \
            "$EXP_DIR/marginals/sample.5.count-filter.$METHOD.lls.npy" \
            > "$EXP_DIR/marginals/sample.5.count-filter.$METHOD.err"
        # train data
        ipython script/compute_marginals.py --  --gpu --batch_size 256 \
            $EXP_DIR "curexp/ptb-train/train.5.count-filter" \
            --method $METHOD --output_filename "train.5.count-filter.$METHOD.lls"
        ipython script/compute_marginal_error.py -- \
            "curexp/ptb-train/train.5.count-filter" \
            "curexp/ptb-train/marginals/train.5.count-filter.count.lls.npy" \
            "$EXP_DIR/marginals/train.5.count-filter.$METHOD.lls.npy" \
            > "$EXP_DIR/marginals/train.5.count-filter.$METHOD.err"
    done
}

# --------------------------------------------------
# Standard training
# --------------------------------------------------
main "curexp/ptb-v" "seqmodel.SeqModel"

# --------------------------------------------------
# Reset state to zero, and MLE unigram
# --------------------------------------------------
main "curexp/ptb-h0-reset0.1-unigram" "seqmodel.UnigramSeqModelH"

# --------------------------------------------------
# Reset state to training variables, and MLE unigram
# --------------------------------------------------
main "curexp/ptb-h-reset0.1-unigram" "seqmodel.UnigramSeqModelH"
