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
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import os
import sys
from functools import partial
from dataclasses import dataclass, field
from typing import Optional

#  from torch.optim import Adam
#  from torch.optim.lr_scheduler import LambdaLR
import datasets
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BertModel,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from cao_align.cao_data import MAX_SENTENCE_LENGTH
from cao_align.utils import (
    DataCollatorForCaoAlignment,
    SizedMultiDataset,
    MultiDataset,
    DataCollatorForCaoMLMAlignment,
    tokenize_function_for_parallel,
    tokenize_function_for_unlabeled,
)
from cao_align.cao_model import (
    BertForCaoAlign,
    BertForCaoAlignMLM,
    BertForLinerLayearAlign,
)
from cao_align.cao_model import CaoTrainer, UnsupervisedTrainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
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
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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
    include_clssep: bool = field(
        default=True,
        metadata={"help": "Use [CLS] and [SEP] in the alignment"},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_mode: Optional[str] = field(
        default='supervised',
        metadata={"help": "Options: supervised, mining"}
    )
    dataset_name: Optional[str] = field(
        default='./cao_align/cao_data.py',
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="es-en,bg-en,fr-en,de-en,el-en,ne-en",
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    mining_dataset_name: Optional[str] = field(
        default='./cao_align/text_data.py',
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    mining_language_pairs: Optional[str] = field(
        default="ne-en,en-ne",
        metadata={"help": "The configuration name of the unsupervised dataset"
                  "to use (via the datasets library). These sets are used as"
                  "the source languages for mining."}
    )
    #  train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    #  validation_file: Optional[str] = field(
    #      default=None,
    #      metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    #  )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    #  validation_split_percentage: Optional[int] = field(
    #      default=5,
    #      metadata={
    #          "help": "The percentage of the train set used as validation set in case there's no validation split"
    #      },
    #  )
    max_seq_length: Optional[int] = field(
        default=MAX_SENTENCE_LENGTH,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
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
    max_mining_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Unsupervised data can be large. Only read n from the head."
        },
    )
    do_mlm: bool = field(default=False, metadata={"help": "Whether to Align+MLM or just Align"})
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    src_mlm_weight: float = field(
        default=0.01, metadata={"help": "Weight of source language MLM loss. 0.0 to turn off."}
    )
    trg_mlm_weight: float = field(
        default=0.01, metadata={"help": "Weight of target language MLM loss. 0.0 to turn off."}
    )

    def __post_init__(self):
        assert not self.do_mlm or self.src_mlm_weight != 0.0 or self.trg_mlm_weight != 0.0, \
            'Either source or target MLM weight should be none-zero!'

        self.mining_language_pairs = [
            item.split('-')
            for item in self.mining_language_pairs.split(',')
        ]

        if self.max_train_samples == -1:
            self.max_train_samples = None
        if self.max_eval_samples == -1:
            self.max_eval_samples = None
        if self.max_mining_samples == -1:
            self.max_mining_samples = None


@dataclass
class MyTrainingArguments(TrainingArguments):
    detailed_logging: bool = field(
        default=False,
        metadata={
            "help": "Detailed logging."
        },
    )
    do_test: bool = field(default=False, metadata={"help": "Whether to run eval on the test set."})
    align_method: Optional[str] = field(
        default='full',
        metadata={
            "help": "Options: full, linear"
        },
    )
    mining_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size for mining. None to use per_device_train_batch_size"
        },
    )
    mining_threshold: float = field(
        default=0.5,
        metadata={"help": "What word pairs to use with mining training"}
    )
    mining_threshold_max: float = field(
        default=100.0,
        metadata={"help": "What word pairs to use with mining training. Filter"
                  "too similar words which are often punctuations, numbers"
                  "or words from the same language due to code switch."}
    )
    mining_k: int = field(
        default=1,
        metadata={"help": "Number of words (above threshold) to mine for each source word"}
    )
    mining_sample_per_step: Optional[int] = field(
        default=1000,
        metadata={
            "help": "Mining can take long on large corpora. Sample data for"
            "each mining step."
        },
    )
    mining_method: Optional[str] = field(
        default='intersection',
        metadata={
            "help": "Options: forward, intersection"
        },
    )
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
    metric_for_best_model: Optional[str] = field(
        default='eval_avg_acc',
        metadata={"help": "The metric to use to compare two different models."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.mining_batch_size is None:
            self.mining_batch_size = self.per_device_train_batch_size

        if self.mining_sample_per_step == -1:
            self.mining_sample_per_step = None

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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    return model_args, data_args, training_args


def get_model_components(model_args, data_args, training_args):
    assert not data_args.do_mlm or training_args.align_method == 'full', 'Currently not supported!'
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if not data_args.do_mlm:
            if training_args.align_method == 'full':
                model = BertForCaoAlign.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                )
            elif training_args.align_method == 'linear':
                num_langs = len(
                    {
                        lang
                        for pair in data_args.dataset_config_name.split(',')
                        for lang in pair.split('-')
                    } | {
                        lang
                        for pair in data_args.mining_language_pairs
                        for lang in pair
                    }
                )
                model = BertForLinerLayearAlign.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    num_languages=num_langs,
                )
            else:
                raise f'Align method not supported: {training_args.align_method}'
        else:
            model = BertForCaoAlignMLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                src_mlm_weight=data_args.src_mlm_weight,
                trg_mlm_weight=data_args.trg_mlm_weight,
            )

        model_base = BertModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    else:
        logger.info("Training new model from scratch")
        if not data_args.do_mlm:
            if training_args.align_method == 'full':
                model = BertForCaoAlign.from_config(config)
            elif training_args.align_method == 'linear':
                num_langs = len(
                    {
                        lang
                        for pair in data_args.dataset_config_name.split(',')
                        for lang in pair.split('-')
                    } | {
                        lang
                        for pair in data_args.mining_language_pairs
                        for lang in pair
                    }
                )
                model = BertForLinerLayearAlign.from_config(
                    config,
                    num_languages=num_langs,
                )
            else:
                raise f'Align method not supported: {training_args.align_method}'
        else:
            model = BertForCaoAlignMLM.from_config(
                config,
                src_mlm_weight=data_args.src_mlm_weight,
                trg_mlm_weight=data_args.trg_mlm_weight,
            )
        model_base = BertModel.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    model_base.resize_token_embeddings(len(tokenizer))
    for param in model_base.parameters():
        param.requires_grad = False

    return tokenizer, model, model_base, last_checkpoint


def get_parallel_datasets(data_args, model_args, training_args, tokenizer, model):
    # Downloading and loading a dataset
    #  raw_datasets = load_dataset(
    #      data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
    #  )
    config_names = data_args.dataset_config_name.split(',')
    load_train = 'train'
    load_validation = 'validation'
    load_test = 'test'
    if data_args.max_train_samples is not None:
        load_train = f'train[:{data_args.max_train_samples}]'
    if data_args.max_eval_samples is not None:
        load_train = f'validation[:{data_args.max_eval_samples}]'

    raw_datasets = {
        config_name: load_dataset(
            data_args.dataset_name,
            config_name,
            cache_dir=model_args.cache_dir,
            split={
                'train': load_train,
                'validation': load_validation,
                'test': load_test,
            },
        )
        for config_name in config_names
    }

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets[config_names[0]]["train"].column_names
    else:
        column_names = raw_datasets[config_names[0]]["validation"].column_names

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Tokenizing data
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = {
            k: v.map(
                partial(tokenize_function_for_parallel, tokenizer),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )
            for k, v in raw_datasets.items()
    }

    train_dataset = SizedMultiDataset({v: k["train"] for v, k in tokenized_datasets.items()})
    eval_dataset = SizedMultiDataset({v: k["validation"] for v, k in tokenized_datasets.items()})
    test_dataset = SizedMultiDataset({v: k["test"] for v, k in tokenized_datasets.items()})

    # Data collator
    # This one will take care converting lists to tensors
    if not data_args.do_mlm:
        data_collator = DataCollatorForCaoAlignment(
            tokenizer=tokenizer,
            max_length=model.bert.embeddings.position_embeddings.num_embeddings,
            include_clssep=model_args.include_clssep,
        )
    else:
        data_collator = DataCollatorForCaoMLMAlignment(
            tokenizer=tokenizer,
            max_length=model.bert.embeddings.position_embeddings.num_embeddings,
            include_clssep=model_args.include_clssep,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of_8=False,
        )

    return train_dataset, eval_dataset, test_dataset, data_collator


def get_mining_datasets(data_args, model_args, tokenizer, model):
    config_names = {lang for item in data_args.mining_language_pairs for lang in item}
    load_train = 'train'
    if data_args.max_mining_samples is not None:
        load_train = f'train[:{data_args.max_mining_samples}]'

    raw_datasets = {
        config_name: load_dataset(
            data_args.mining_dataset_name,
            config_name,
            cache_dir=model_args.cache_dir,
            split={
                'train': load_train,
            },
        )
        for config_name in config_names
    }

    return MultiDataset({v: k["train"] for v, k in raw_datasets.items()})


def get_trainer(model_args, data_args, training_args, tokenizer, model,
                model_base, train_dataset, eval_dataset, mining_dataset,
                data_collator):
    # Initialize our Trainer
    #  optimizer = Adam([param for param in model.parameters() if
    #                    param.requires_grad], lr=training_args.learning_rate,
    #                   betas=(training_args.adam_beta1, training_args.adam_beta2), eps=1e-9)
    #
    #  def lr_lambda(current_step: int):
    #      if current_step < training_args.warmup_steps:
    #          return float(current_step) / float(max(1, training_args.warmup_steps))
    #      return 1.0
    #
    #  scheduler = LambdaLR(
    #      optimizer,
    #      lr_lambda,
    #  )
    #  trainer = Trainer(
    callbacks = []
    if training_args.early_stopping_patience >= 0:
        callbacks.append(EarlyStoppingCallback(
            training_args.early_stopping_patience,
            training_args.early_stopping_threshold,
        ))

    if data_args.data_mode == 'supervised':
        trainer = CaoTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            bert_base=model_base,
            include_clssep=model_args.include_clssep,
            callbacks=callbacks,
            #  optimizers=(optimizer, scheduler),
        )
    elif data_args.data_mode == 'mining':
        trainer = UnsupervisedTrainer(
            model=model,
            args=training_args,
            train_dataset=mining_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            bert_base=model_base,
            include_clssep=model_args.include_clssep,
            language_pairs=data_args.mining_language_pairs,
            callbacks=callbacks,
            #  optimizers=(optimizer, scheduler),
        )

    return trainer


def run(data_args, training_args, trainer, last_checkpoint, train_dataset,
        eval_dataset, test_dataset):
    # Training
    if training_args.do_train:
        if training_args.do_eval:
            # avaluate the base model
            metrics = trainer.evaluate()
            trainer.log(metrics)

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Validation ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Test
    if training_args.do_test:
        logger.info("*** Testing ***")

        metrics = trainer.evaluate(test_dataset)

        max_test_samples = len(test_dataset)
        metrics["test_samples"] = max_test_samples

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


def main():
    model_args, data_args, training_args = setup()
    tokenizer, model, model_base, last_checkpoint = get_model_components(
        model_args,
        data_args,
        training_args,
    )

    train_dataset, eval_dataset, test_dataset, data_collator = get_parallel_datasets(
        data_args,
        model_args,
        training_args,
        tokenizer,
        model,
    )

    mining_dataset = None
    if data_args.data_mode == 'mining':
        mining_dataset = get_mining_datasets(data_args, model_args,
                                             tokenizer, model)

    trainer = get_trainer(model_args, data_args, training_args, tokenizer,
                          model, model_base, train_dataset, eval_dataset,
                          mining_dataset, data_collator)

    run(data_args, training_args, trainer, last_checkpoint, train_dataset,
        eval_dataset, test_dataset)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
