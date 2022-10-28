import collections
import os
import numpy as np
import datasets
from itertools import zip_longest
from align.alignment_utils import keep_1to1


_DESCRIPTION = """\
Dataset containing parallel sentence pairs and their alignments.
"""

_CITATION = """\
"""

_DATA_URL = "https://www.cis.uni-muenchen.de/~hangyav/data/parallel_data.zip"

# Tuple that describes a single pair of files with matching translations.
# language_to_file is the map from language (2 letter string: example 'en')
# to the file path in the extracted directory.
# TODO
TranslateData = collections.namedtuple("TranslateData", ["url", "language_to_file"])


LANGUAGE_PATHS = {
    'es-en': ('es-en.tokens', 'es-en.alignments'),
    'bg-en': ('bg-en.tokens', 'bg-en.alignments'),
    'fr-en': ('fr-en.tokens', 'fr-en.alignments'),
    'de-en': ('de-en.tokens', 'de-en.alignments'),
    'el-en': ('el-en.tokens', 'el-en.alignments'),
    'ne-en': ('ne-en.tokens', 'ne-en.alignments'),
    'am-en': ('am-en.tokens', 'am-en.alignments'),
    'mi-en': ('mi-en.tokens', 'mi-en.alignments'),
    'ml-en': ('ml-en.tokens', 'ml-en.alignments'),
    'sd-en': ('sd-en.tokens', 'sd-en.alignments'),
    'si-en': ('si-en.tokens', 'si-en.alignments'),
    'sw-en': ('sw-en.tokens', 'sw-en.alignments'),
}

# Order of data is TEST, DEV, TRAIN
# -1 mean all of the rest
LANGUAGE_SENTENCES_NUMBERS = {
    'es-en': (1024, 1024, -1),
    'bg-en': (1024, 1024, -1),
    'fr-en': (1024, 1024, -1),
    'de-en': (1024, 1024, -1),
    'el-en': (1024, 1024, -1),
    'ne-en': (1024, 1024, -1),
    'am-en': (1024, 1024, -1),
    'mi-en': (1024, 1024, -1),
    'ml-en': (1024, 1024, -1),
    'sd-en': (1024, 1024, -1),
    'si-en': (1024, 1024, -1),
    'sw-en': (1024, 1024, -1),
}

MAX_SENTENCE_LENGTH = 128


class ParallelDataConfig(datasets.BuilderConfig):

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
        super(ParallelDataConfig, self).__init__(
            name=name,
            description=description,
            version=datasets.Version("1.0.1", ""),
            **kwargs,
        )

        # Validate language pair.
        assert name in LANGUAGE_PATHS, f'Unsupported language pair: {name}'
        self.language_pair = language_pair


class ParallelData(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ParallelDataConfig(
            language_pair=("es", "en"),
        ),
        ParallelDataConfig(
            language_pair=("fr", "en"),
        ),
        ParallelDataConfig(
            language_pair=("de", "en"),
        ),
        ParallelDataConfig(
            language_pair=("bg", "en"),
        ),
        ParallelDataConfig(
            language_pair=("el", "en"),
        ),
        ParallelDataConfig(
            language_pair=("ne", "en"),
        ),
        ParallelDataConfig(
            language_pair=("am", "en"),
        ),
        ParallelDataConfig(
            language_pair=("mi", "en"),
        ),
        ParallelDataConfig(
            language_pair=("ml", "en"),
        ),
        ParallelDataConfig(
            language_pair=("sd", "en"),
        ),
        ParallelDataConfig(
            language_pair=("si", "en"),
        ),
        ParallelDataConfig(
            language_pair=("sw", "en"),
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
                "source_language": datasets.Value("string"),
                "target_language": datasets.Value("string"),
                }),
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
                    "language_pair": self.config.language_pair,
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepaths": paths,
                    "start_idx": LANGUAGE_SENTENCES_NUMBERS[name][0],
                    "num_lines": LANGUAGE_SENTENCES_NUMBERS[name][1],
                    "language_pair": self.config.language_pair,
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": paths,
                    "start_idx": 0,
                    "num_lines": LANGUAGE_SENTENCES_NUMBERS[name][0],
                    "language_pair": self.config.language_pair,
                }
            ),
        ]

    def _generate_examples(self, filepaths, start_idx, num_lines, language_pair):
        for idx, item in enumerate(generate_samples(
            filepaths,
            start_idx,
            num_lines,
            language_pair
        )):
            yield idx, item


def generate_samples(filepaths, start_idx, num_lines, language_pair):
    with open(filepaths[0]) as f_sents, open(filepaths[1]) as f_align:
        num = 0
        for i, (sents, align) in enumerate(zip_longest(f_sents, f_align)):
            assert sents is not None and align is not None

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

                yield {
                    "source": sent1,
                    "target": sent2,
                    "alignment": align_lst,
                    "source_language": language_pair[0],
                    "target_language": language_pair[1],
                }
