#!/bin/bash

METHODS="init"

main () {
    EXP_DIR=$1
    MODEL_CLASS=$2
    VOCAB=$3

    source experiments/marginal/_decode_trace.sh "data/wikitext-2" $EXP_DIR $MODEL_CLASS

    ipython script/compute_marginals.py --  --gpu --batch_size 256 \
        $EXP_DIR "$EXP_DIR/marginals/sample.5.count-filter" \
        --method "count" --output_filename "sample.5.count-filter.count.lls" \
        --vocab_path $VOCAB

    for METHOD in $METHODS ; do
        # sample data
        ipython script/compute_marginals.py --  --gpu --batch_size 256 \
            $EXP_DIR "$EXP_DIR/marginals/sample.5.count-filter" \
            --method $METHOD --output_filename "sample.5.count-filter.$METHOD.lls" \
            --vocab_path $VOCAB
        ipython script/compute_marginal_error.py -- \
            "$EXP_DIR/marginals/sample.5.count-filter" \
            "$EXP_DIR/marginals/sample.5.count-filter.count.lls.npy" \
            "$EXP_DIR/marginals/sample.5.count-filter.$METHOD.lls.npy" \
            > "$EXP_DIR/marginals/sample.5.count-filter.$METHOD.err"
        # train data
        ipython script/compute_marginals.py --  --gpu --batch_size 256 \
            $EXP_DIR "curexp/wt2-train/train.5.count-filter" \
            --method $METHOD --output_filename "train.5.count-filter.$METHOD.lls" \
            --vocab_path $VOCAB
        ipython script/compute_marginal_error.py -- \
            "curexp/wt2-train/train.5.count-filter" \
            "curexp/wt2-train/marginals/train.5.count-filter.count.lls.npy" \
            "$EXP_DIR/marginals/train.5.count-filter.$METHOD.lls.npy" \
            > "$EXP_DIR/marginals/train.5.count-filter.$METHOD.err"
    done
}

# --------------------------------------------------
# Standard training
# --------------------------------------------------
main "curexp/wt2-v-med" "seqmodel.SeqModel" "data/wikitext-2/vocab.txt"

# --------------------------------------------------
# Reset state to zero, and MLE unigram
# --------------------------------------------------
main "curexp/wt2-h0-reset0.05-unigram-med" "seqmodel.UnigramSeqModelH" "data/wikitext-2/vocab.txt"

# --------------------------------------------------
# Reset state to training variables, and MLE unigram
# --------------------------------------------------
main "curexp/wt2-h-reset0.05-unigram-med" "seqmodel.UnigramSeqModelH" "data/wikitext-2/vocab.txt"
