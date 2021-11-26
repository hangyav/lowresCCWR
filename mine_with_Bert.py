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

import logging
import os
import sys
from functools import partial
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from cao_align.cao_data import MAX_SENTENCE_LENGTH
from cao_align.utils import (
    DataCollatorForUnlabeledData,
    tokenize_function_for_unlabeled,
)
from cao_align.cao_model import BertForCaoAlign

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

    model_name_or_path: str = field(
        metadata={
            "help": "The model checkpoint for weights initialization."
        },
    )
    threshold: float = field(
        metadata={"help": "Minimum word pair similarity."},
    )
    k: int = field(
        default=1,
        metadata={"help": "Top-k pairs to mine for each word token in the source corpus."},
    )
    batch_size: int = field(
        default=16,
        metadata={"help": ""},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    device: str = field(
        default='cuda:0',
        metadata={
            "help": "cpu, cuda:0, etc"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    src_dataset_path: str = field(
        metadata={"help": "The path of the source dataset to use."}
    )
    trg_dataset_path: str = field(
        metadata={"help": "The path of the target dataset to use."}
    )
    output: str = field(
        metadata={"help": "Path to save mined word pairs"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
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


def main():
    # See all possible arguments by passing the --help flag to this script.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
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

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = BertForCaoAlign.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(model_args.device)
    model.eval()

    src_dataset = load_dataset(
        'text',
        data_files={
            'test': data_args.src_dataset_path
        }
    )
    src_dataset = src_dataset.map(
        partial(tokenize_function_for_unlabeled, tokenizer),
        batched=True,
        num_proc=1,
        #  remove_columns=src_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on the source dataset",
    )
    src_dataset = src_dataset['test']

    trg_dataset = load_dataset(
        'text',
        data_files={
            'test': data_args.trg_dataset_path
        }
    )
    trg_dataset = trg_dataset.map(
        partial(tokenize_function_for_unlabeled, tokenizer),
        batched=True,
        num_proc=1,
        #  remove_columns=src_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on the target dataset",
    )
    trg_dataset = trg_dataset['test']

    data_collator = DataCollatorForUnlabeledData(
        tokenizer=tokenizer,
        max_length=model.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=False,
    )

    output = model.mine_word_pairs(
        src_dataset,
        trg_dataset,
        model_args.threshold,
        data_collator,
        k=model_args.k,
        batch_size=model_args.batch_size,
    )

    def _process(align_lst, src_dataset, trg_dataset, output):
        if align_lst is None or len(align_lst) == 0:
            return

        src_sent_id = align_lst[0][0]
        trg_sent_id = align_lst[0][2]
        src_sent = src_dataset['text'][src_sent_id]
        trg_sent = trg_dataset['text'][trg_sent_id]
        alignments = [f'{align[1]}-{align[3]}' for align in align_lst]
        scores = [f'{align[4]}' for align in align_lst]

        print('{} ||| {} ||| {} ||| {}'.format(
            src_sent,
            trg_sent,
            ' '.join(alignments),
            ' '.join(scores),
        ),
            file=output
        )

    with open(data_args.output, 'w') as fout:
        tmp_lst = None
        last_sent_pair = None
        for item in output:
            curr_sent_pair = (item[0], item[2])

            if last_sent_pair != curr_sent_pair:
                _process(tmp_lst, src_dataset, trg_dataset, fout)
                tmp_lst = list()
                last_sent_pair = curr_sent_pair

            tmp_lst.append(item)

        _process(tmp_lst, src_dataset, trg_dataset, fout)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
