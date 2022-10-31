#!/bin/bash

if [ $# -lt 5 ]; then
    echo "Usage: $0 <src_tokenized_file.txt> <trg_tokenized_file.txt> <src_language> <trg_language> <output> [model_dir <align_method:linear|full>]"
    exit 1
fi

set -x

PYTHON="${PYTHON:-python}"
SRC_INPUT=$1
TRG_INPUT=$2
SRC_LANGUAGE=$3
TRG_LANGUAGE=$4
OUTPUT=$5

MODEL="bert-base-multilingual-cased"
METHOD="full"
if [ $# -eq 6 ]; then
    set +x
    echo
    echo "If model_dir is specified then align_method is also needed!"
    exit 1
fi
if [ $# -ge 7 ]; then
    MODEL=$6
    METHOD=$7
fi


$PYTHON mine_with_Bert.py \
    --model_name_or_path $MODEL \
    --align_method $METHOD \
    --src_dataset_path $SRC_INPUT \
    --trg_dataset_path $TRG_INPUT \
    --src_language $SRC_LANGUAGE \
    --trg_language $TRG_LANGUAGE \
    --output $OUTPUT \
    --threshold 0.2 \
    --threshold_max 100.0 \
    --k 1 \
    --batch_size 16 \
    --preprocessing_num_workers 20 \
    --mine_method forward
