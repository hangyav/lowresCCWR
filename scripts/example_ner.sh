#!/bin/bash

set -x

PYTHON="${PYTHON:-python}"
INPUT_FILE=$1
OUTPUT_DIR='output'
MODEL_DIR=$OUTPUT_DIR/NER
SRC_LANGUAGE='en'
TRG_LANGUAGE='ne'

MODEL="bert-base-multilingual-cased"
if [ $# -ge 1 ]; then
    MODEL=$1
fi

mkdir -p $OUTPUT_DIR

# COMMENTS:
# --dataset_config_name: <train_language>,<validation_language>,<test_language>
#
# Validation and test results are in $MODEL_DIR/eval_results.json and
# $MODEL_DIR/predict_results.json respectively. We report "*_context_avg" in
# our paper.
#
# Remove --do_train for evaluation only.

$PYTHON run_ner.py \
    --model_name_or_path $MODEL \
    --dataset_name wikiann \
    --dataset_config_name ${SRC_LANGUAGE},${TRG_LANGUAGE},${TRG_LANGUAGE} \
    --languages $TRG_LANGUAGE,$SRC_LANGUAGE \
    --logging_steps 1000 \
    --eval_steps 1000 \
    --early_stopping_patience 10 \
    --evaluation_strategy epoch \
    --num_train_epochs 100 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-08 \
    --warmup_steps 1000 \
    --save_strategy epoch \
    --save_steps 1000 \
    --save_total_limit 3 \
    --overwrite_output_dir \
    --output_dir $MODEL_DIR \
    --logging_dir $MODEL_DIR/log \
    --preprocessing_num_workers 1 \
    --metric_for_best_model eval_f1 \
    --freeze_bert_core 1 \
    --do_train \
    --do_eval \
    --do_predict
