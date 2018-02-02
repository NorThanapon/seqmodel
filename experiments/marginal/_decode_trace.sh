#!/bin/bash

DATADIR=$1
EXPDIR=$2
MODEL_CLASS="$3"
OUTDIR="$EXPDIR/marginals"
COMMONOPT="
$DATADIR tmp --gpu --log_level debug --batch_size 20 --seq_len 1
--load_checkpoint "$EXPDIR/checkpoint/best"
--load_model_opt $EXPDIR/model_opt.json
--model_class $MODEL_CLASS
"

mkdir -p $OUTDIR

echo "Decoding..."
python main_lm.py "decode" $COMMONOPT \
    --out:decode --decode:add_sampling --decode:outpath "$OUTDIR/sample.txt"

echo "Building n-gram count..."
ngram-count -order 9 -text "$OUTDIR/sample.txt" -write "$OUTDIR/sample.9.count"
ngram-count -order 5 -text "$OUTDIR/sample.txt" -write "$OUTDIR/sample.5.count"
ngram-count -order 3 -text "$OUTDIR/sample.txt" -write "$OUTDIR/sample.3.count"
ngram-count -order 1 -text "$OUTDIR/sample.txt" -write "$OUTDIR/unigrams.count"
python script/filter_ngram.py "$OUTDIR/sample.5.count" -1 2
wc -lw < "$OUTDIR/sample.txt" | awk '{s+=$1+$2} END {print s}' > "$OUTDIR/total_tokens.txt"
cd "$OUTDIR"
ln -sf "sample.5.count-filter" "ngrams.count"
cd -

echo "Generating trace from training data..."
python main_lm.py "eval" $COMMONOPT \
    --eval_file "train.txt" --trace_state_filename "train.txt-trace.npy"
mv "tmp/train.txt-trace.npy" "$OUTDIR/train.txt-trace.npy"
