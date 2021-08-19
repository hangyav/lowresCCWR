import argparse
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertTokenizer, BertForPreTraining
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)


def extend_bert(model_path, train_data, vocab_size):
    logger.warning('Loading...')
    # Do not use fast tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    tok_unk = tokenizer.unk_token
    tok_cls = tokenizer.cls_token
    tok_sep = tokenizer.sep_token
    tok_pad = tokenizer.pad_token
    tok_mask = tokenizer.mask_token

    new_tokenizer = Tokenizer(WordPiece(unk_token=tok_unk))

    norm_lst = list()
    norm_lst.append(NFD())
    if tokenizer.do_lower_case:
        norm_lst.append(Lowercase())

        if tokenizer.init_kwargs['strip_accents'] is not False:
            norm_lst.append(StripAccents())

    elif tokenizer.init_kwargs['strip_accents']:
        norm_lst.append(StripAccents())
    new_tokenizer.normalizer = normalizers.Sequence(norm_lst)

    new_tokenizer.pre_tokenizer = Whitespace()
    new_tokenizer.post_processor = TemplateProcessing(
        single=f'{tok_cls} $A {tok_sep}',
        pair=f'{tok_cls} $A {tok_sep} $B:1 {tok_sep}:1',
        special_tokens=[
            (f'{tok_cls}', 1),
            (f'{tok_sep}', 2),
        ],
    )

    trainer = WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=[
            tok_unk, tok_cls, tok_sep, tok_pad, tok_mask]
    )
    logger.warning('Training new tokenizer...')
    new_tokenizer.train([train_data], trainer)

    # this makes the tokenizer slow
    #  num = tokenizer.add_tokens(list(new_tokenizer.get_vocab().keys()))
    num = 0
    for w in tqdm(list(new_tokenizer.get_vocab().keys()), desc="Adding tokens"):
        if w not in tokenizer.vocab:
            tokenizer.vocab[w] = len(tokenizer.vocab)
            num += 1
    logger.warning(f'Added {num} new tokens')

    model = BertForPreTraining.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Model to load')
    parser.add_argument('--train_data', required=True,
                        help='Data to build new vocabulary')
    parser.add_argument('--model_type', default='bert',
                        help='Data to build new vocabulary')
    parser.add_argument('--vocab_size', default=10000, type=int,
                        help='Number of added vocab entries')
    parser.add_argument('--output', help='Folder to save to', required=True)
    args = parser.parse_args()

    model_path = args.model_path
    train_data = args.train_data
    model_type = args.model_type.lower()
    vocab_size = args.vocab_size

    assert model_type in ['bert'], f'{model_type} model is not yet supported'

    if model_type in ['bert']:
        tokenizer, model = extend_bert(model_path, train_data, vocab_size)
    else:
        assert False, "We shouldn't get here!"

    logger.warning('Saving...')
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
