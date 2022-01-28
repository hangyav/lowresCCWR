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
from tqdm import tqdm
import numpy as np
from flashtext import KeywordProcessor
from functools import partial
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data.dataloader import DataLoader
import datasets

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from cao_align.cao_data import MAX_SENTENCE_LENGTH
from cao_align.utils import (
    DataCollatorForUnlabeledData,
    tokenize_function_for_unlabeled,
    save_embeddings,
)
from cao_align.cao_model import (
    BertForCaoAlign,
    BertForLinerLayearAlign,
)

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
    align_method: Optional[str] = field(
        default='full',
        metadata={
            "help": "Sets the model type. Options: full, linear"
        },
    )
    batch_size: int = field(
        default=128,
        metadata={
            "help": "Batch size."
        },
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
    no_cuda: bool = field(
        default=False,
        metadata={"help": "Do not use CUDA even when it is available"}
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

    dataset_path: str = field(
        default=None,
        metadata={"help": "The path of the dataset to use."}
    )
    language: str = field(
        default=None,
        metadata={"help": "The language of the dataset. Mostly important for"
                  " language specific alignment models, e.g., LinearLayer model."}
    )
    vocabulary_path: str = field(
        default=None,
        metadata={"help": "The path of the vocabulary to use. One word per line."}
    )
    output_path: str = field(
        default=None,
        metadata={"help": "The file to save embeddings in."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached datasets"}
    )
    max_seq_length: Optional[int] = field(
        default=MAX_SENTENCE_LENGTH,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_samples_per_word: Optional[int] = field(
        default=100,
        metadata={
            "help": "Number of embeddings to average per word."
        },
    )

    def __post_init__(self):
        if self.max_samples_per_word == -1:
            self.max_samples_per_word = None


def setup():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(0)

    return model_args, data_args


def get_model_components(model_args, data_args):
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

    if model_args.align_method == 'full':
        model = BertForCaoAlign.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.align_method == 'linear':
        model = BertForLinerLayearAlign.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            languages={data_args.language},
        )
    else:
        raise f'Align method not supported: {model_args.align_method}'

    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def get_raw_dataset_components(data_args):
    with open(data_args.vocabulary_path) as fin:
        vocabulary = [w.strip() for w in fin]

    keyword_processor = KeywordProcessor(case_sensitive=True)
    for w in vocabulary:
        keyword_processor.add_keyword(w)

    vocabulary_map = dict()
    sentences = list()
    stop = False

    with open(data_args.dataset_path) as fin:
        for line in tqdm(fin, desc='Loading dataset'):
            line = line.strip()
            kws = set(keyword_processor.extract_keywords(line))

            if len(kws) > 0:
                idx = len(sentences)
                sentences.append(line)
                sent = line.split()

                for kw in kws:
                    vocabulary_map.setdefault(kw, list()).append((
                        idx,
                        sent.index(kw),
                    ))
                    if len(vocabulary_map[kw]) >= data_args.max_samples_per_word:
                        keyword_processor.remove_keyword(kw)
                        if len(keyword_processor.get_all_keywords()) == 0:
                            stop = True
                            break
            if stop:
                break

    logger.info('Words with no sentences: {}'.format(
        len(vocabulary) - len(vocabulary_map.keys())
    ))
    return vocabulary, vocabulary_map, sentences


def get_dataset(sentences, tokenizer, model, data_args):
    dataset = datasets.Dataset.from_dict({
        'text': [
            text for text in sentences
        ],
        'language': [data_args.language] * len(sentences)
    })

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

    dataset = dataset.map(
        partial(
            tokenize_function_for_unlabeled,
            tokenizer,
        ),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=dataset.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    data_collator = DataCollatorForUnlabeledData(
        tokenizer=tokenizer,
        max_length=model.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=False,
    )

    return dataset, data_collator


def run(model, dataset, data_collator, vocabulary_map, vocabulary, data_args,
        model_args):
    reversed_vocab_map = dict()
    for k, v in vocabulary_map.items():
        # k: word
        for vv in v:
            # vv[0]: sentence id
            # vv[1]: word position in sentence
            reversed_vocab_map.setdefault(vv[0], list()).append((vv[1], k))

    data_loader = DataLoader(
        dataset,
        batch_size=model_args.batch_size,
        collate_fn=data_collator,
        pin_memory=model.bert.device == 'cuda:0',
    )

    batch_offset = 0
    emb_dict = dict()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Building representations'):
            features = model.process_sentences(batch, False, False)

            for i in range(features.shape[0]):
                sent_idx = i + batch_offset
                for w_pos, w in reversed_vocab_map[sent_idx]:
                    emb_dict.setdefault(w, list()).append(
                        features[i][w_pos].detach().to('cpu').numpy()
                    )

            batch_offset += model_args.batch_size

    res = {
        k: np.mean(v, axis=0)
        for k, v in emb_dict.items()
    }

    logger.info('Saving anchors to: {}'.format(data_args.output_path))
    save_embeddings(res, data_args.output_path, vocabulary)


def main():
    model_args, data_args = setup()
    tokenizer, model = get_model_components(
        model_args,
        data_args,
    )
    model.eval()
    if not model_args.no_cuda and torch.cuda.is_available():
        model = model.to('cuda:0')

    vocabulary, vocabulary_map, sentences = get_raw_dataset_components(data_args)
    dataset, data_collator = get_dataset(sentences, tokenizer, model, data_args)

    run(model, dataset, data_collator, vocabulary_map, vocabulary, data_args,
        model_args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
