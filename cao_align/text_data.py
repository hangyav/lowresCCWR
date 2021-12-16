import os
import datasets


_DESCRIPTION = """\
Dataset containing plain sentences. We have this only for ease of use.
"""

_CITATION = """\
"""

_DATA_URL = "https://www.cis.uni-muenchen.de/~hangyav/data/ccwe_text_data.zip"

# Tuple that describes a single pair of files with matching translations.
# language_to_file is the map from language (2 letter string: example 'en')
# to the file path in the extracted directory.
# TODO
#  TranslateData = collections.namedtuple("TranslateData", ["url", "language_to_file"])


LANGUAGE_PATHS = {
    'en': 'enwiki.tok.sample',
    'de': 'dewiki.tok.sample',
    'ne': 'wiki-ne.tok.sample',
}

MAX_SENTENCE_LENGTH = 128
MIN_SENTENCE_LENGTH = 5


class TextConfig(datasets.BuilderConfig):

    def __init__(self, language, **kwargs):
        description = f'Dataset from {language}'
        super(TextConfig, self).__init__(
            name=language,
            description=description,
            version=datasets.Version("1.0.1", ""),
            **kwargs,
        )

        # Validate language pair.
        assert language in LANGUAGE_PATHS, f'Unsupported language: {language}'
        self.language = language


class Text(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        TextConfig(language="en"),
        TextConfig(language="de"),
        TextConfig(language="ne"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "text": datasets.Value("string"),
                "language": datasets.Value("string"),
            }),
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(_DATA_URL)

        name = self.config.language
        path = LANGUAGE_PATHS[name]
        path = os.path.join(dl_dir, path)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": path,
                    "language": name
                }
            ),
        ]

    def _generate_examples(self, filepath, language):
        with open(filepath) as fin:
            idx = 0
            for sent in fin:
                sent = sent.strip()

                length = len(sent.split())
                if length > MAX_SENTENCE_LENGTH or length < MIN_SENTENCE_LENGTH:
                    continue

                yield idx, {
                    "text": sent,
                    "language": language
                }
                idx += 1
