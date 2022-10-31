#!/bin/bash

set -x

PYTHON="${PYTHON:-python}"
INPUT_FILE=$1
OUTPUT_DIR='output'
MODEL_DIR=$OUTPUT_DIR/aligned_model

MODEL="bert-base-multilingual-cased"
if [ $# -ge 1 ]; then
    MODEL=$1
fi

mkdir -p $OUTPUT_DIR

# COMMENTS:
# Set --dataset_dir (for parallel) and --mining_dataset_dir (for monolingual)
# to point to the folders where tokenized files are saved (useful if you'd like
# to use custom data or language pairs instead of downloading the ones used in
# our paper automatically). Look at --dataset_sentences_pattern_name,
# --dataset_alignments_pattern_name and --mining_dataset_pattern_name for
# naming conventions for a given language pair (--dataset_config_name)
#
# Validation and test results are in $MODEL_DIR/eval_results.json and
# $MODEL_DIR/test_results.json respectively. We report "*_context_avg" in our
# paper.
#
# Remove --do_train for evaluation only.

$PYTHON run_align.py \
    --data_mode mining \
    --model_name_or_path $MODEL \
    --dataset_config_name ne-en \
    --mining_language_pairs ne-en,en-ne \
    --logging_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --early_stopping_patience -1 \
    --max_steps 5000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 256 \
    --mining_src_batch_size 64 \
    --mining_trg_batch_size 64 \
    --gradient_accumulation_steps 6 \
    --learning_rate 5e-05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-09 \
    --warmup_steps 1000 \
    --save_steps 5000 \
    --save_total_limit 5 \
    --overwrite_output_dir \
    --output_dir $MODEL_DIR \
    --logging_dir $MODEL_DIR/log \
    --detailed_logging 0 \
    --do_train \
    --do_eval \
    --do_test \
    --max_train_samples -1 \
    --max_eval_samples -1 \
    --max_mining_samples -1 \
    --src_mining_sample_per_step 1000 \
    --trg_mining_sample_per_step 1000 \
    --dataloader_num_workers 0 \
    --preprocessing_num_workers 20 \
    --do_mlm 0 \
    --src_mlm_weight 0.0 \
    --trg_mlm_weight 0.0 \
    --mining_threshold 0.2 \
    --mining_threshold_max 100.0 \
    --mining_k 1 \
    --mining_method forward \
    --align_method linear \
    --faiss_index_str none
