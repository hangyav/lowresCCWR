#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <untokenized_file.txt>"
    exit 1
fi

set -x

PYTHON="${PYTHON:-python}"
INPUT_FILE=$1
OUTPUT_DIR='output'
BASE_MODEL_DIR=$OUTPUT_DIR/mbert_ne_extended_base
MLM_MODEL_DIR=$OUTPUT_DIR/mbert_ne_extended

mkdir -p $OUTPUT_DIR

$PYTHON bert_vocabulary_extender.py \
    --model_path bert-base-multilingual-cased \
    --model_type bert \
    --train_data $INPUT_FILE \
    --vocab_size 10000 \
    --output $BASE_MODEL_DIR

$PYTHON run_mlm.py \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 4 \
    --max_steps 100000 \
    --warmup_steps 10000 \
    --eval_steps 5000 \
    --save_steps 5000 \
    --max_seq_length 512 \
    --model_name_or_path $BASE_MODEL_DIR \
    --train_file $INPUT_FILE \
    --overwrite_output_dir  \
    --output_dir $MLM_MODEL_DIR \
    --logging_dir $MLM_MODEL_DIR/log \
    --do_train \
    --do_eval \
    --validation_split_percentage 5 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 5000
