import collections
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
import os
import numpy as np
import datasets
from cao_align.multilingual_alignment import keep_1to1
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import _collate_batch


_DESCRIPTION = """\
Dataset containing parallel sentence pairs and their alignments.
"""

_CITATION = """\
"""

_DATA_URL = "https://www.cis.uni-muenchen.de/~hangyav/data/cao_data.zip"

# Tuple that describes a single pair of files with matching translations.
# language_to_file is the map from language (2 letter string: example 'en')
# to the file path in the extracted directory.
# TODO
TranslateData = collections.namedtuple("TranslateData", ["url", "language_to_file"])


LANGUAGE_PATHS = {
    'es-en': ('europarl-v7.es-en.token.clean', 'europarl-v7.es-en.intersect'),
    'bg-en': ('europarl-v7.bg-en.token.clean', 'europarl-v7.bg-en.intersect'),
    'fr-en': ('europarl-v7.fr-en.token.clean', 'europarl-v7.fr-en.intersect'),
    'de-en': ('europarl-v7.de-en.token.clean', 'europarl-v7.de-en.intersect'),
    'el-en': ('europarl-v7.el-en.token.clean', 'europarl-v7.el-en.intersect'),
    #  'ne-en': ('europarl-v7.ne-en.token.clean', 'europarl-v7.ne-en.intersect'),
    'ne-en': ('final.data.clean.2.txt', 'final.data.clean.2.intersect'),
}

# Order of data is TEST, DEV, TRAIN
# -1 mean all
LANGUAGE_SENTENCES_NUMBERS = {
    'es-en': (1024, 1024, -1),
    'bg-en': (1024, 1024, -1),
    'fr-en': (1024, 1024, -1),
    'de-en': (1024, 1024, -1),
    'el-en': (1024, 1024, -1),
    'ne-en': (1024, 1024, -1),

}

MAX_SENTENCE_LENGTH = 128


class CaoConfig(datasets.BuilderConfig):

    def __init__(self, language_pair=(None, None), **kwargs):
        """BuilderConfig for FLoRes.
        Args:
            for the `datasets.features.text.TextEncoder` used for the features feature.
          language_pair: pair of languages that will be used for translation. Should
            contain 2-letter coded strings. First will be used at source and second
            as target in supervised mode. For example: ("fr", "en").
          **kwargs: keyword arguments forwarded to super.
        """
        name = "%s-%s" % (language_pair[0], language_pair[1])

        description = ("Dataset from %s to %s") % (language_pair[0], language_pair[1])
        super(CaoConfig, self).__init__(
            name=name,
            description=description,
            version=datasets.Version("1.0.0", ""),
            **kwargs,
        )

        # Validate language pair.
        assert name in LANGUAGE_PATHS, f'Unsupported language pair: {name}'
        self.language_pair = language_pair


class Cao(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CaoConfig(
            language_pair=("es", "en"),
        ),
        CaoConfig(
            language_pair=("fr", "en"),
        ),
        CaoConfig(
            language_pair=("de", "en"),
        ),
        CaoConfig(
            language_pair=("bg", "en"),
        ),
        CaoConfig(
            language_pair=("el", "en"),
        ),
        CaoConfig(
            language_pair=("ne", "en"),
        ),
    ]

    def _info(self):
        source, target = self.config.language_pair
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "source": datasets.Value("string"),
                "target": datasets.Value("string"),
                "alignment": datasets.Sequence(
                    datasets.Sequence(datasets.Value('int32'))
                ),
                #  "alignment": datasets.Array2D(shape=(-1, 2), dtype=int),
                }),
            #  supervised_keys=("source", "target", "alignment"),
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)

        name = '-'.join(self.config.language_pair)
        paths = LANGUAGE_PATHS[name]
        paths = (
                os.path.join(dl_dir, paths[0]),
                os.path.join(dl_dir, paths[1]),
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": paths,
                    "start_idx": LANGUAGE_SENTENCES_NUMBERS[name][0] + LANGUAGE_SENTENCES_NUMBERS[name][1],
                    "num_lines": LANGUAGE_SENTENCES_NUMBERS[name][2],
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": paths,
                    "start_idx": LANGUAGE_SENTENCES_NUMBERS[name][0],
                    "num_lines": LANGUAGE_SENTENCES_NUMBERS[name][1],
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": paths,
                    "start_idx": 0,
                    "num_lines": LANGUAGE_SENTENCES_NUMBERS[name][0],
                }
            ),
        ]

    def _generate_examples(self, filepaths, start_idx, num_lines):
        with open(filepaths[0]) as f_sents, open(filepaths[1]) as f_align:
            idx = 0
            num = 0
            for i, (sents, align) in enumerate(zip(f_sents, f_align)):
                sents = sents.strip()
                align = align.strip()
                if start_idx <= i:
                    if num == num_lines:
                        break
                    num += 1

                    sent1, sent2 = sents.split(' ||| ')
                    if len(sent1.split()) > MAX_SENTENCE_LENGTH or len(sent2.split()) > MAX_SENTENCE_LENGTH:
                        continue

                    align_lst = np.array([
                        list(map(int, pair.split('-')))
                        for pair in align.split()
                    ])
                    align_lst = keep_1to1(align_lst)

                    yield idx, {
                        "source": sent1,
                        "target": sent2,
                        "alignment": align_lst,
                    }
                    idx += 1


@dataclass
class DataCollatorForCaoAlignment:

    tokenizer: PreTrainedTokenizerBase
    max_length: int

    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        max_src_len = max([len(item['src_input_ids']) for item in examples])
        max_trg_len = max([len(item['trg_input_ids']) for item in examples])

        src_input_ids = self._handle_input_ids(
                examples,
                'src_input_ids',
                max_src_len,
                self.tokenizer.pad_token_id,
        )
        trg_input_ids = self._handle_input_ids(
                examples,
                'trg_input_ids',
                max_trg_len,
                self.tokenizer.pad_token_id,
        )

        src_special_word_masks = self._handle_special_word_masks(
                examples,
                'src_special_word_masks',
                src_input_ids,
        )
        trg_special_word_masks = self._handle_special_word_masks(
                examples,
                'trg_special_word_masks',
                trg_input_ids,
        )

        src_attention_masks = (src_special_word_masks != 2).type(torch.IntTensor)
        trg_attention_masks = (trg_special_word_masks != 2).type(torch.IntTensor)

        src_word_ids_lst = self._to_list(examples, 'src_word_ids_lst')
        trg_word_ids_lst = self._to_list(examples, 'trg_word_ids_lst')
        alignment = self._to_list(examples, 'alignment')

        return {
            'src_input_ids': src_input_ids,
            'trg_input_ids': trg_input_ids,
            'src_attention_masks': src_attention_masks,
            'trg_attention_masks': trg_attention_masks,
            'src_word_ids_lst': src_word_ids_lst,
            'trg_word_ids_lst': trg_word_ids_lst,
            'src_special_word_masks': src_special_word_masks,
            'trg_special_word_masks': trg_special_word_masks,
            'alignment': alignment,
        }

    def _handle_input_ids(self, examples, col, max_len, default_value):
        input_ids = torch.zeros((len(examples), max_len), dtype=int)
        input_ids = input_ids.fill_(default_value)

        for i, example in enumerate(examples):
            input_ids[i, :len(example[col])] = torch.tensor(example[col], dtype=int)

        return input_ids

    def _handle_special_word_masks(self, examples, col, input_ids):
        special_word_masks = torch.zeros_like(input_ids).fill_(2)

        for i, example in enumerate(examples):
            special_word_masks[i, :len(example[col])] = torch.tensor(example[col], dtype=int)

        return special_word_masks

    def _to_list(self, examples, col):
        return [example[col] for example in examples]


class MultiDataLoader():

    def __init__(self, datasets):
        self.datasets = datasets
        # this is ugly
        self.dataset = self

    def __len__(self):
        return sum([len(d.dataset) for d in self.datasets])

    def __iter__(self):
        iters = [d.__iter__() for d in self.datasets]
        while True:
            stopped = set()
            for it in iters:
                try:
                    yield next(it)
                except StopIteration:
                    stopped.add(it)
            for it in stopped:
                iters.remove(it)
            if len(iters) == 0:
                return
