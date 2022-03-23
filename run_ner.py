#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

import datasets
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric, DatasetDict

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from cao_align.cao_model import (
    BertForTokenClassification,
    BertForCaoAlign,
    BertForLinerLayearAlign,
    BertForPretrainedLinearLayerAlign,
    BertForCaoAlignMLM,
)
from cao_align.utils import (
    tokenizer_function_for_ner,
    DataCollatorForNER,
)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    languages: Optional[str] = field(
        default=None,
        metadata={
            "help": "Only when training from scratch. Space separated language"
            "ids for linear and pretrained."
        }
    )
    pretrained_alignments: Optional[str] = field(
        default=None,
        metadata={"help": "Alignments built with eg. VecMap. Format: [<lang>=<path>,...]"},
    )
    freeze_bert_core: bool = field(
        default=True,
        metadata={
            "help": "Whether to freeze model core parameters."
        },
    )

    def __post_init__(self):
        if self.languages is not None:
            self.languages = set(self.languages.split(','))

        if self.pretrained_alignments is not None:
            self.pretrained_alignments = {
                item.split('=')[0]: item.split('=')[1]
                for item in self.pretrained_alignments.split(',')
            }

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default='wikiann', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset"
                                " to use (via the datasets library). Can be"
                                " comma separated for train,validation,test"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()

        if self.label_all_tokens:
            raise NotImplementedError('Lable all tokens is not supported!')

        if self.dataset_config_name is not None:
            self.dataset_config_name = self.dataset_config_name.split(',')


@dataclass
class MyTrainingArguments(TrainingArguments):
    early_stopping_patience: int = field(
        default=-1,
        metadata={
            "help": ">=0 to set early stopping"
        },
    )
    early_stopping_threshold: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Denote how much the specified metric must improve to satisfy early stopping conditions."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.early_stopping_patience >= 0:
            self.load_best_model_at_end = True


def setup():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device},"
        f"n_gpu: {training_args.n_gpu} distributed training:"
        f" {bool(training_args.local_rank != -1)}, 16-bits training:"
        f" {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # processing dataset_config_name
    assert 0 < len(data_args.dataset_config_name) <= 3, 'Train[, Validation][, Test]'
    while len(data_args.dataset_config_name) < 3:
        data_args.dataset_config_name.append(data_args.dataset_config_name[-1])

    return model_args, data_args, training_args


def get_datasets(data_args, model_args, training_args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training
    # and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the
    # hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the
    # first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only
    # one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = DatasetDict({
            split: load_dataset(
                data_args.dataset_name,
                config_name,
                cache_dir=model_args.cache_dir,
                split=split,
            )
            for split, config_name in zip(['train', 'validation', 'test'], data_args.dataset_config_name)
        })
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files,
                                    cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from
    # files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to
    # go through the dataset to get the unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    return raw_datasets, text_column_name, label_column_name, num_labels, label_to_id, label_list


def get_model_components(model_args, data_args, training_args, num_labels,
                         label_to_id):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) \
            and training_args.do_train \
            and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists"
                " and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
                " To avoid this behavior, change the `--output_dir` or add"
                " `--overwrite_output_dir` to train from scratch."
            )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={i: l for l, i in label_to_id.items()},
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = (model_args.tokenizer_name
                              if model_args.tokenizer_name
                              else model_args.model_name_or_path)
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    arch = config.architectures[0]
    if arch == BertForTokenClassification.__name__:
        # TODO this is not nice. Should refactor and put parameters to config.
        # But it will do for now
        sub_arch = config.subarchitecture

        if sub_arch == BertForCaoAlign.__name__:
            aligned_model = BertForCaoAlign(config)
        else:
            languages = set(config.language_layers) | set(data_args.dataset_config_name)
            if sub_arch == BertForLinerLayearAlign.__name__:
                aligned_model = BertForLinerLayearAlign(
                    config,
                    languages,
                )
            elif sub_arch == BertForPretrainedLinearLayerAlign.__name__:
                aligned_model = BertForPretrainedLinearLayerAlign(
                    config,
                    languages,
                    {},
                )
            else:
                raise f'Architecture not supported: {sub_arch}'

        model = BertForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            bert=aligned_model,
            freeze_bert_core=model_args.freeze_bert_core,
        )
    else:
        params = dict()

        if arch == BertForLinerLayearAlign.__name__:
            languages = set(model_args.languages) | set(data_args.dataset_config_name)
            aligned_model = BertForLinerLayearAlign.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                languages=languages,
            )
            params['language_layers'] = list(languages)
        elif arch == BertForCaoAlignMLM.__name__:
            sub_arch = config.subarchitecture
            if sub_arch == BertForCaoAlign.__name__:
                aligned_model = BertForCaoAlign(config)
            else:
                languages = set(model_args.languages) | set(data_args.dataset_config_name)
                params['language_layers'] = list(languages)
                if sub_arch == BertForLinerLayearAlign.__name__:
                    aligned_model = BertForLinerLayearAlign(
                        config,
                        languages,
                    )
                elif sub_arch == BertForPretrainedLinearLayerAlign.__name__:
                    aligned_model = BertForPretrainedLinearLayerAlign(
                        config,
                        languages,
                        {},
                    )
                else:
                    raise f'Architecture not supported: {sub_arch}'

                model = BertForCaoAlignMLM.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    src_mlm_weight=0.00001,
                    trg_mlm_weight=0.00001,
                    bert=aligned_model,
                )
                aligned_model = model.bert
        elif arch == BertForCaoAlign.__name__ and model_args.pretrained_alignments is not None:
            languages = set(model_args.languages) | set(data_args.dataset_config_name)
            lang_map = {
                lang: np.load(path)
                for lang, path in model_args.pretrained_alignments.items()
            }
            aligned_model = BertForPretrainedLinearLayerAlign.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                languages=languages,
                language_mappings=lang_map,
            )
            params['language_layers'] = list(languages)
        else:
            # BertForCaoAlign or just Bert in general
            aligned_model = BertForCaoAlign.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        model = BertForTokenClassification(config, freeze_bert_core=model_args.freeze_bert_core)
        model.set_bert(aligned_model)
        params['subarchitecture'] = type(aligned_model).__name__
        model.config.update(params)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast"
            " tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks"
            " to find the model types that meet this requirement"
        )

    return tokenizer, model, last_checkpoint


def get_processed_datasets(data_args, training_args, tokenizer, model,
                           raw_datasets, text_column_name, label_column_name,
                           label_to_id):
    max_seq_length = tokenizer.model_max_length
    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                partial(
                    tokenizer_function_for_ner,
                    tokenizer,
                    text_column_name,
                    label_column_name,
                    label_to_id,
                    max_seq_length,
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                partial(
                    tokenizer_function_for_ner,
                    tokenizer,
                    text_column_name,
                    label_column_name,
                    label_to_id,
                    max_seq_length,
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                partial(
                    tokenizer_function_for_ner,
                    tokenizer,
                    text_column_name,
                    label_column_name,
                    label_to_id,
                    max_seq_length,
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    data_collator = DataCollatorForNER(
        tokenizer=tokenizer,
        max_length=model.bert.bert.embeddings.position_embeddings.num_embeddings,
        label_pad_token_id=-100,
    )

    return train_dataset, eval_dataset, predict_dataset, data_collator


def run(data_args, training_args, tokenizer, model, last_checkpoint,
        train_dataset, eval_dataset, predict_dataset, data_collator,
        label_list):
    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    callbacks = []
    if training_args.early_stopping_patience >= 0:
        callbacks.append(EarlyStoppingCallback(
            training_args.early_stopping_patience,
            training_args.early_stopping_threshold,
        ))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")


def main():
    model_args, data_args, training_args = setup()

    raw_datasets, text_column_name, label_column_name, num_labels, label_to_id, label_list = get_datasets(
        data_args,
        model_args,
        training_args,
    )

    tokenizer, model, last_checkpoint = get_model_components(
        model_args,
        data_args,
        training_args,
        num_labels,
        label_to_id,
    )

    train_dataset, eval_dataset, predict_dataset, data_collator = get_processed_datasets(
        data_args,
        training_args,
        tokenizer,
        model,
        raw_datasets,
        text_column_name,
        label_column_name,
        label_to_id,
    )

    run(data_args, training_args, tokenizer, model, last_checkpoint,
        train_dataset, eval_dataset, predict_dataset, data_collator,
        label_list)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
