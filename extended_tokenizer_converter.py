import argparse
from transformers import AutoTokenizer
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)


def add_special_tokens_to_normal_vocab(tokenizer):
    """
    This is weird but if there are tokens in added_tokens.json the tokenizer
    becomes super slow: https://github.com/huggingface/tokenizers/issues/615
    """
    if len(tokenizer.added_tokens_encoder) > 0:
        for w, i in tqdm(tokenizer.added_tokens_encoder.items()):
            assert w not in tokenizer.vocab, f'Encoder word {w}:{i} already contained'
            tokenizer.vocab[w] = i
        tokenizer.added_tokens_encoder = dict()
        tokenizer.added_tokens_decoder = dict()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Model to load')
    parser.add_argument('--output', help='Folder to save to', required=True)
    args = parser.parse_args()

    model_path = args.model_path
    output = args.output

    logger.warning('Loading...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    add_special_tokens_to_normal_vocab(tokenizer)

    logger.warning('Saving...')
    tokenizer.save_pretrained(output)
