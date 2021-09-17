import pytest
import torch
import run_cao as rc
from cao_align import cao_model as cm
from transformers import AutoTokenizer


@pytest.fixture
def tokenizer_bert_multilingual_cased():
    return AutoTokenizer.from_pretrained('bert-base-multilingual-cased')


@pytest.mark.parametrize('examples,expected', [
    (
        [
            'I like beer .',
            'Ich mag Weißbier und Kartoffel .',
        ],
        {
            'tokens': [
                ['[CLS]', 'I', 'like', 'beer', '.', '[SEP]'],
                ['[CLS]', 'Ich', 'mag', 'Weiß', '##bier', 'und', 'Kar',
                 '##tof', '##fel', '.', '[SEP]'],
            ],
            'word_ids_lst': [
                [[0], [1], [2], [3], [4], [5]],
                [[0], [1], [2], [3, 4], [5], [6, 7, 8], [9], [10]],
            ],
            'special_word_masks': [
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
        }
    ),
])
def test_tokenize_function_per_input(examples, expected,
                                     tokenizer_bert_multilingual_cased):
    output = rc.tokenize_function_per_input(tokenizer_bert_multilingual_cased,
                                            examples)
    assert [
            tokenizer_bert_multilingual_cased.convert_ids_to_tokens(ids)
            for ids in output['input_ids']
        ] == expected['tokens']
    assert output['word_ids_lst'] == expected['word_ids_lst']
    assert output['special_word_masks'] == expected['special_word_masks']


@pytest.mark.parametrize('features,word_ids_lsts,special_word_masks,'
                         'include_clssep,expected', [
    (
        torch.tensor([
            [[0.0]*5, [1.0]*5, [2.0]*5, [-1.0]*5, [-1.0]*5, [-1.0]*5],
            [[0.0]*5, [1.0]*5, [1.5]*5, [2.0]*5, [2.5]*5, [3.0]*5],
        ]),
        [
            [[0], [1], [2]],
            [[0], [1, 2], [3, 4], [5]],
        ],
        [
            [1, 0, 1, 2, 2, 2],
            [1, 0, 0, 0, 0, 1],
        ],
        True,
        torch.tensor([
            [[0.0]*5, [1.0]*5, [2.0]*5, [0.0]*5],
            [[0.0]*5, [1.5]*5, [2.5]*5, [3.0]*5],
        ]),
    ),
    (
        torch.tensor([
            [[0.0]*5, [1.0]*5, [2.0]*5, [-1.0]*5, [-1.0]*5, [-1.0]*5],
            [[0.0]*5, [1.0]*5, [1.5]*5, [2.0]*5, [2.5]*5, [3.0]*5],
        ]),
        [
            [[0], [1], [2]],
            [[0], [1, 2], [3, 4], [5]],
        ],
        [
            [1, 0, 1, 2, 2, 2],
            [1, 0, 0, 0, 0, 1],
        ],
        False,
        torch.tensor([
            [[1.0]*5, [0.0]*5],
            [[1.5]*5, [2.5]*5],
        ]),
    ),
])
def test_SubwordToTokenStrategyLast(features, word_ids_lsts,
                                    special_word_masks, include_clssep,
                                    expected):
    strategy = cm.SubwordToTokenStrategyLast()
    output = strategy.process(features, word_ids_lsts, special_word_masks,
                              include_clssep)

    assert output.tolist() == expected.tolist()
