import os
import datasets


_DESCRIPTION = """\
Dataset containing plain sentences.
"""

_CITATION = """\
"""

_DATA_URL = "https://www.cis.uni-muenchen.de/~hangyav/data/ccwe_text_data.zip"


LANGUAGE_PATHS = {
    'en': 'enwiki.tok.sample',
    'de': 'dewiki.tok.sample',
    'ne': 'wiki-ne.tok.sample',
    'am': 'amwiki.tok.sample',
    'ml': 'mlwiki.tok.sample',
    'si': 'siwiki.tok.sample',
    'sw': 'swwiki.tok.sample',
    'mi': 'miwiki.tok.sample',
    'sd': 'sdwiki.tok.sample',
    'hi': 'hiwiki.tok.sample',
}

MAX_SENTENCE_LENGTH = 128
MIN_SENTENCE_LENGTH = 5


class TextConfig(datasets.BuilderConfig):

    def __init__(self, language, **kwargs):
        description = f'Dataset from {language}'
        super(TextConfig, self).__init__(
            name=language,
            description=description,
            version=datasets.Version("1.0.2", ""),
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
        TextConfig(language="am"),
        TextConfig(language="ml"),
        TextConfig(language="si"),
        TextConfig(language="sw"),
        TextConfig(language="mi"),
        TextConfig(language="sd"),
        TextConfig(language="hi"),
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
        for idx, sent in enumerate(generate_samples(filepath)):
            yield idx, {
                "text": sent,
                "language": language
            }


def generate_samples(path):
    with open(path) as fin:
        for sent in fin:
            sent = sent.strip()

            length = len(sent.split())
            if length > MAX_SENTENCE_LENGTH or length < MIN_SENTENCE_LENGTH:
                continue

            yield sent
