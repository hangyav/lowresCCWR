import os
import pytest
import torch
from functools import partial
import copy
import tempfile

import run_align as rc
from align import model as cm
from align import utils as cu
from transformers import AutoTokenizer, BertModel, AutoConfig
from datasets import Dataset, DatasetDict


@pytest.fixture
def tokenizer_bert_multilingual_cased():
    return AutoTokenizer.from_pretrained('bert-base-multilingual-cased')


@pytest.fixture
def bert_base():
    model_base = BertModel.from_pretrained('bert-base-multilingual-cased')
    for param in model_base.parameters():
        param.requires_grad = False

    return model_base


@pytest.fixture
def align_bert():
    return cm.BertForFullAlign.from_pretrained('bert-base-multilingual-cased')


@pytest.fixture
def linear_bert():
    return cm.BertForLinerLayearAlign.from_pretrained(
        'bert-base-multilingual-cased',
        languages={'en', 'de'}
    )


@pytest.fixture
def align_mlm_bert():
    model = cm.BertForFullAlignMLM(
        AutoConfig.from_pretrained('bert-base-multilingual-cased'),
        bert=None,
        src_mlm_weight=0.01,
        trg_mlm_weight=0.01,
    )
    model.bert = cm.BertForFullAlign.from_pretrained('bert-base-multilingual-cased')
    return model


@pytest.fixture
def linear_mlm_bert():
    model = cm.BertForFullAlignMLM(
        AutoConfig.from_pretrained('bert-base-multilingual-cased'),
        bert=None,
        src_mlm_weight=0.01,
        trg_mlm_weight=0.01,
    )
    model.bert = cm.BertForLinerLayearAlign.from_pretrained(
        'bert-base-multilingual-cased',
        languages={'en', 'de'}
    )
    return model


@pytest.mark.parametrize('examples,expected', [
    (
        [
            'I like beer .',
            'Ich mag Weißbier und Kartoffel .',
            'I like beer  .',
        ],
        {
            'tokens': [
                ['[CLS]', 'I', 'like', 'beer', '.', '[SEP]'],
                ['[CLS]', 'Ich', 'mag', 'Weiß', '##bier', 'und', 'Kar',
                 '##tof', '##fel', '.', '[SEP]'],
                ['[CLS]', 'I', 'like', 'beer', '[UNK]', '.', '[SEP]'],
            ],
            'word_ids_lst': [
                [[0], [1], [2], [3], [4], [5]],
                [[0], [1], [2], [3, 4], [5], [6, 7, 8], [9], [10]],
                [[0], [1], [2], [3], [4], [5], [6]],
            ],
            'special_word_masks': [
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
            ],
        }
    ),
])
def test_tokenize_function_per_input(examples, expected,
                                     tokenizer_bert_multilingual_cased):
    output = cu.tokenize_function_per_input(tokenizer_bert_multilingual_cased, 512,
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
                                     [[0.0]*5, [1.0]*5, [2.0]*5, [-1.0]
                                         * 5, [-1.0]*5, [-1.0]*5],
                                     [[0.0]*5, [1.0]*5, [1.5]*5,
                                         [2.0]*5, [2.5]*5, [3.0]*5],
                                 ]),
                                 [
                                     [[0], [1], [2]],
                                     [[0], [1, 2], [3, 4], [5]],
                                 ],
                                 torch.tensor([
                                     [1, 0, 1, 2, 2, 2],
                                     [1, 0, 0, 0, 0, 1],
                                 ]),
                                 True,
                                 torch.tensor([
                                     [[0.0]*5, [1.0]*5, [2.0]*5, [0.0]*5],
                                     [[0.0]*5, [1.5]*5, [2.5]*5, [3.0]*5],
                                 ]),
                             ),
                             (
                                 torch.tensor([
                                     [[0.0]*5, [1.0]*5, [2.0]*5, [-1.0]
                                         * 5, [-1.0]*5, [-1.0]*5],
                                     [[0.0]*5, [1.0]*5, [1.5]*5,
                                         [2.0]*5, [2.5]*5, [3.0]*5],
                                 ]),
                                 [
                                     [[0], [1], [2]],
                                     [[0], [1, 2], [3, 4], [5]],
                                 ],
                                 torch.tensor([
                                     [1, 0, 1, 2, 2, 2],
                                     [1, 0, 0, 0, 0, 1],
                                 ]),
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


@pytest.mark.parametrize('alignment,src_special_word_masks,'
                         'trg_special_word_masks,src_word_ids_lst,'
                         'trg_word_ids_lst,expected', [
                             (
                                 [
                                     [(0, 0), (1, 2), (2, 1), (4, 3)],
                                 ],
                                 torch.tensor([
                                     [1, 0, 0, 0, 0, 0, 1],
                                 ]),
                                 torch.tensor([
                                     [1, 0, 0, 0, 0, 1],
                                 ]),
                                 torch.tensor([
                                     [[0], [1], [2], [3], [4], [5], [6]],
                                 ]),
                                 torch.tensor([
                                     [[0], [1], [2], [3], [4], [5]],
                                 ]),
                                 (
                                     torch.tensor([
                                         [0, 1],
                                         [0, 2],
                                         [0, 3],
                                         [0, 5],
                                         # [CLS] and [SEP] at the end of list of
                                         # sample i
                                         [0, 0],
                                         [0, 6],
                                     ]),
                                     torch.tensor([
                                         [0, 1],
                                         [0, 3],
                                         [0, 2],
                                         [0, 4],
                                         [0, 0],
                                         [0, 5],
                                     ]),
                                 ),
                             ),
                         ])
def test_word_alignment(alignment, src_special_word_masks,
                        trg_special_word_masks, src_word_ids_lst,
                        trg_word_ids_lst, expected):
    output = cu.DataCollatorForAlignment.get_aligned_indices(
        alignment,
        True,
        src_special_word_masks,
        trg_special_word_masks,
        src_word_ids_lst,
        trg_word_ids_lst
    )

    assert output[0].tolist() == expected[0].tolist()
    assert output[1].tolist() == expected[1].tolist()


@pytest.mark.parametrize('examples,expected,equals', [
    (
        {
            'source': [
                'I like beer .',
            ],
            'target': [
                'I like beer .',
            ],
            'alignment': [
                [(0, 0), (1, 1), (2, 2), (3, 3)],
            ],
            'source_language': [
                'en',
            ],
            'target_language': [
                'en',
            ]
        },
        {
            'alignment_loss': 0.0,
            'regularization_loss': 0.0,
            'loss': 0.0,
        },
        {
            'alignment_equals': True,
            'regularization_equals': True,
            'equals': True,
        },
    ),
    (
        {
            'source': [
                'I like beer .',
            ],
            'target': [
                'I like beer .',
            ],
            'alignment': [
                [(0, 0), (1, 2), (2, 1), (3, 3)],
            ],
            'source_language': [
                'en',
            ],
            'target_language': [
                'en',
            ]
        },
        {
            'alignment_loss': 0.0,
            'regularization_loss': 0.0,
            'loss': 0.0,
        },
        {
            'alignment_equals': False,
            'regularization_equals': True,
            'equals': False,
        },
    ),
    (
        {
            'source': [
                'I like beer .',
            ],
            'target': [
                'Ich mag Bier .',
            ],
            'alignment': [
                [(0, 0), (1, 1), (2, 2), (3, 3)],
            ],
            'source_language': [
                'en',
            ],
            'target_language': [
                'en',
            ]
        },
        {
            'alignment_loss': 0.0,
            'regularization_loss': 0.0,
            'loss': 0.0,
        },
        {
            'alignment_equals': False,
            'regularization_equals': True,
            'equals': False,
        },
    ),
])
def test_pipeline(examples, expected, equals,
                  bert_base, align_bert, tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
    )
    dataset = Dataset.from_dict(examples)
    dataset = dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    batch = collator(dataset)

    output = align_bert(return_dict=True, bert_base=bert_base, **batch)

    if equals['alignment_equals']:
        assert output['alignment_loss'].item() == expected['alignment_loss']
    else:
        assert output['alignment_loss'].item() != expected['alignment_loss']

    if equals['regularization_equals']:
        assert output['regularization_loss'].item() == expected['regularization_loss']
    else:
        assert output['regularization_loss'].item() != expected['regularization_loss']

    if equals['equals']:
        assert output['loss'].item() == expected['loss']
    else:
        assert output['loss'].item() != expected['loss']


@pytest.mark.parametrize('train,eval', [
    (
        {
            'source': [
                'I like beer .',
                'I like beer .',
            ],
            'target': [
                'I like beer .',
                'Ich mag Bier .'
            ],
            'alignment': [
                [(0, 0), (1, 1), (2, 2), (3, 3)],
                [(0, 0), (1, 1), (2, 2), (3, 3)],
            ],
            'source_language': [
                'en',
                'en',
            ],
            'target_language': [
                'en',
                'en',
            ]
        },
        {
            'source': [
                # needs to have at least k=10 tokens
                'I like water . 5 6 7 8 9 10',
            ],
            'target': [
                'I like water . 5 6 7 8 9 10',
            ],
            'alignment': [
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                 (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)],
            ],
            'source_language': [
                'en',
            ],
            'target_language': [
                'en',
            ]
        },
    ),
])
def test_trainer(train, eval,
                 bert_base, align_bert, tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
    )
    train_dataset = Dataset.from_dict(train)
    train_dataset = train_dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    train_dataset = cu.SizedMultiDataset({'test': train_dataset})
    eval_dataset = Dataset.from_dict(eval)
    eval_dataset = eval_dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    eval_dataset = cu.SizedMultiDataset({'test': eval_dataset})
    trainer = cm.SupervisedTrainer(
        model=align_bert,
        args=rc.MyTrainingArguments(
            detailed_logging=True,
            output_dir='/tmp',
            logging_steps=1,
            save_steps=1000,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            evaluation_strategy='steps',
            eval_steps=2,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer_bert_multilingual_cased,
        data_collator=collator,
        bert_base=bert_base,
        include_clssep=rc.ModelArguments().include_clssep,
    )
    trainer.train()
    trainer.evaluate()


@pytest.mark.parametrize('train,eval', [
    (
        {
            'source': [
                'I like beer .',
                'I like beer .',
            ],
            'target': [
                'I like beer .',
                'Ich mag Bier .'
            ],
            'alignment': [
                [(0, 0), (1, 1), (2, 2), (3, 3)],
                [(0, 0), (1, 1), (2, 2), (3, 3)],
            ],
            'source_language': [
                'en',
                'en',
            ],
            'target_language': [
                'en',
                'en',
            ],
        },
        {
            'source': [
                # needs to have at least k=10 tokens
                'I like water . 5 6 7 8 9 10',
            ],
            'target': [
                'I like water . 5 6 7 8 9 10',
            ],
            'alignment': [
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                 (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)],
            ],
            'source_language': [
                'en',
            ],
            'target_language': [
                'en',
            ],
        },
    ),
])
def test_trainer_mlm(train, eval,
                     bert_base, align_mlm_bert,
                     tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForMLMAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_mlm_bert.bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
        mlm_probability=0.9,
        pad_to_multiple_of_8=False,
    )
    train_dataset = Dataset.from_dict(train)
    train_dataset = train_dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    train_dataset = cu.SizedMultiDataset({'test': train_dataset})
    eval_dataset = Dataset.from_dict(eval)
    eval_dataset = eval_dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    eval_dataset = cu.SizedMultiDataset({'test': eval_dataset})
    trainer = cm.SupervisedTrainer(
        model=align_mlm_bert,
        args=rc.MyTrainingArguments(
            detailed_logging=True,
            output_dir='/tmp',
            logging_steps=1,
            save_steps=1000,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            evaluation_strategy='steps',
            eval_steps=2,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer_bert_multilingual_cased,
        data_collator=collator,
        bert_base=bert_base,
        include_clssep=rc.ModelArguments().include_clssep,
    )
    trainer.train()
    trainer.evaluate()


mlm_pipeline_data = [
    (
        {
            'source': [
                'I like beer .',
            ],
            'target': [
                'I like beer .',
            ],
            'alignment': [
                [(0, 0), (1, 1), (2, 2), (3, 3)],
            ],
            'source_language': [
                'en',
            ],
            'target_language': [
                'en',
            ],
        },
        {
            'alignment_loss': 0.0,
            'regularization_loss': 0.0,
            'combined_alignment_loss': 0.0,
            'loss': 0.0,
        },
        {
            'alignment_equals': True,
            'regularization_equals': True,
            'combined_equals': True,
            'equals': False,
        },
    ),
    (
        {
            'source': [
                'I like beer .',
            ],
            'target': [
                'I like beer .',
            ],
            'alignment': [
                [(0, 0), (1, 2), (2, 1), (3, 3)],
            ],
            'source_language': [
                'en',
            ],
            'target_language': [
                'en',
            ],
        },
        {
            'alignment_loss': 0.0,
            'regularization_loss': 0.0,
            'combined_alignment_loss': 0.0,
            'loss': 0.0,
        },
        {
            'alignment_equals': False,
            'regularization_equals': True,
            'combined_equals': False,
            'equals': False,
        },
    ),
    (
        {
            'source': [
                'I like beer .',
            ],
            'target': [
                'Ich mag Bier .',
            ],
            'alignment': [
                [(0, 0), (1, 1), (2, 2), (3, 3)],
            ],
            'source_language': [
                'en',
            ],
            'target_language': [
                'en',
            ],
        },
        {
            'alignment_loss': 0.0,
            'regularization_loss': 0.0,
            'combined_alignment_loss': 0.0,
            'loss': 0.0,
        },
        {
            'alignment_equals': False,
            'regularization_equals': True,
            'combined_equals': False,
            'equals': False,
        },
    ),
]


def _mlm_pipeline(examples, expected, equals,
                  bert_base, model, tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForMLMAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=model.bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
        mlm_probability=1.0,
        pad_to_multiple_of_8=False,
    )
    dataset = Dataset.from_dict(examples)
    dataset = dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    batch = collator(dataset)

    assert (batch['src_input_ids'] != batch['src_mlm_input_ids']).type(
        torch.IntTensor).sum().item() > 0
    assert (batch['trg_input_ids'] != batch['trg_mlm_input_ids']).type(
        torch.IntTensor).sum().item() > 0

    output = model(return_dict=True, bert_base=bert_base, **batch)

    if equals['alignment_equals']:
        assert output['alignment_loss'].item() == expected['alignment_loss']
    else:
        assert output['alignment_loss'].item() != expected['alignment_loss']

    if equals['regularization_equals']:
        assert output['regularization_loss'].item() == expected['regularization_loss']
    else:
        assert output['regularization_loss'].item() != expected['regularization_loss']

    if equals['combined_equals']:
        assert output['combined_alignment_loss'].item() == expected['combined_alignment_loss']
    else:
        assert output['combined_alignment_loss'].item() != expected['combined_alignment_loss']

    if equals['equals']:
        assert output['loss'].item() == expected['loss']
    else:
        assert output['loss'].item() != expected['loss']

    assert output.src_mlm_output.loss.item() > 0.0
    assert output.trg_mlm_output.loss.item() > 0.0


@pytest.mark.parametrize('examples,expected,equals', mlm_pipeline_data)
def test_mlm_pipeline(examples, expected, equals,
                      bert_base, align_mlm_bert, tokenizer_bert_multilingual_cased):
    _mlm_pipeline(examples, expected, equals, bert_base, align_mlm_bert,
                  tokenizer_bert_multilingual_cased)


@pytest.mark.parametrize('examples,expected,equals', mlm_pipeline_data)
def test_linear_mlm_pipeline(examples, expected, equals,
                             bert_base, linear_mlm_bert,
                             tokenizer_bert_multilingual_cased):
    _mlm_pipeline(examples, expected, equals, bert_base, linear_mlm_bert,
                  tokenizer_bert_multilingual_cased)


@pytest.mark.parametrize('examples,expected', [
    (
        [
            'I like beer .',
            'Ich mag Weißbier und Kartoffel .',
        ],
        [
            '[CLS] I like beer . [SEP]'.split(),
            '[CLS] Ich mag Weißbier und Kartoffel . [SEP]'.split(),
        ],
    ),
])
def test_detokenize(examples, expected, tokenizer_bert_multilingual_cased):
    output = cu.detokenize(
        tokenizer_bert_multilingual_cased(examples)['input_ids'],
        tokenizer_bert_multilingual_cased,
    )

    assert output == expected


@pytest.mark.parametrize('src,trg,threshold,k,expected', [
    (
        {
            'text': [
                'I like beer .',
                'Du learnst Deutsch .',
                'I like beer .',
                'ThisTokenIsSplit to BPEs'
            ],
            'language': [
                'en',
                'de',
                'en',
                'en',
            ],
        },
        {
            'text': [
                'I like beer .',
                'Du learnst Deutsch .',
                'ThisTokenIsSplit to BPEs'
            ],
            'language': [
                'en',
                'en',
                'en',
            ],
        },
        0.0,
        1,
        [
            (0, 0, 0, 0), (0, 1, 0, 1), (0, 2, 0, 2), (0, 3, 0, 3),
            (1, 0, 1, 0), (1, 1, 1, 1), (1, 2, 1, 2), (1, 3, 1, 3),
            (2, 0, 0, 0), (2, 1, 0, 1), (2, 2, 0, 2), (2, 3, 0, 3),
            (3, 0, 2, 0), (3, 1, 2, 1), (3, 2, 2, 2),
        ],
    ),
    (
        {
            'text': [
                'Apfel'
            ],
            'language': [
                'en',
            ],
        },
        {
            'text': [
                'This is a dummy sentence',
                'This is a dummy sentence',
                'This is a dummy sentence',
                'This is a dummy sentence',
                'Der ist ein Apfel',
                'This is a dummy sentence',
                'Apfel ist der beste',
            ],
            'language': [
                'en',
                'en',
                'en',
                'en',
                'en',
                'en',
                'en',
            ],
        },
        0.6,
        2,
        [
            (0, 0, 4, 3), (0, 0, 6, 0),
        ],
    ),
])
def test_mining(src, trg, threshold, k, expected,
                align_bert, tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForUnlabeledData(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=False,
    )
    src_dataset = Dataset.from_dict(src)
    src_dataset = src_dataset.map(
        partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=src_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    trg_dataset = Dataset.from_dict(trg)
    trg_dataset = trg_dataset.map(
        partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=trg_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )

    output = align_bert.mine_word_pairs(src_dataset, trg_dataset, threshold,
                                        collator, k=k, src_batch_size=1,
                                        trg_batch_size=1)
    output_idxs = [item[:4] for item in output]

    assert output_idxs == expected
    assert all([item[4] is not None for item in output])


@pytest.mark.parametrize('src,trg,threshold,k,expected', [
    (
        {
            'text': [
                'I like beer .',
                'Du learnst Deutsch .',
                'I like beer .',
                'ThisTokenIsSplit to BPEs'
            ],
            'language': [
                'en',
                'de',
                'en',
                'en',
            ],
        },
        {
            'text': [
                'I like beer .',
                'Du learnst Deutsch .',
                'ThisTokenIsSplit to BPEs',
            ] + ['This is a dummy sentence'] * (500-3),
            'language': [
                'en',
                'en',
                'en',
            ] + ['en'] * (500-3),
        },
        0.0,
        1,
        [
            (0, 0, 0, 0), (0, 1, 0, 1), (0, 2, 0, 2), (0, 3, 0, 3),
            (1, 0, 1, 0), (1, 1, 1, 1), (1, 2, 1, 2), (1, 3, 1, 3),
            (2, 0, 0, 0), (2, 1, 0, 1), (2, 2, 0, 2), (2, 3, 0, 3),
            (3, 0, 2, 0), (3, 1, 2, 1), (3, 2, 2, 2),
        ],
    ),
])
def test_mining_faiss(src, trg, threshold, k, expected,
                      align_bert, tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForUnlabeledData(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=False,
    )
    src_dataset = Dataset.from_dict(src)
    src_dataset = src_dataset.map(
        partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=src_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    trg_dataset = Dataset.from_dict(trg)
    trg_dataset = trg_dataset.map(
        partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=trg_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )

    output = align_bert.mine_word_pairs(src_dataset, trg_dataset, threshold,
                                        collator, k=k, src_batch_size=500,
                                        trg_batch_size=500,
                                        faiss_index_str='Flat')
    output_idxs = [item[:4] for item in output]

    assert output_idxs == expected
    assert all([item[4] is not None for item in output])


@pytest.mark.parametrize('src,trg,threshold,k,expected', [
    (
        {
            'text': [
                'I like beer .',
                'Du learnst Deutsch .',
                'I like beer .',
                'ThisTokenIsSplit to BPEs'
            ],
            'language': [
                'en',
                'de',
                'en',
                'en',
            ],
        },
        {
            'text': [
                'I like beer .',
                'Du learnst Deutsch .',
                'ThisTokenIsSplit to BPEs'
            ],
            'language': [
                'en',
                'en',
                'en',
            ],
        },
        0.0,
        1,
        [
            (0, 0, 0, 0), (0, 1, 0, 1), (0, 2, 0, 2), (0, 3, 0, 3),
            (1, 0, 1, 0), (1, 1, 1, 1), (1, 2, 1, 2), (1, 3, 1, 3),
            (2, 0, 0, 0), (2, 1, 0, 1), (2, 2, 0, 2), (2, 3, 0, 3),
            (3, 0, 2, 0), (3, 1, 2, 1), (3, 2, 2, 2),
        ],
    ),
    (
        {
            'text': [
                'Apfel'
            ],
            'language': [
                'de',
            ],
        },
        {
            'text': [
                'This is a dummy sentence',
                'This is a dummy sentence',
                'This is a dummy sentence',
                'This is a dummy sentence',
                'Der ist ein Apfel',
                'This is a dummy sentence',
                'Apfel ist der beste',
            ],
            'language': [
                'en',
                'en',
                'en',
                'en',
                'en',
                'en',
                'en',
            ],
        },
        0.6,
        2,
        [
            (0, 0, 4, 3), (0, 0, 6, 0),
        ],
    ),
])
def test_intersection_mining(src, trg, threshold, k, expected,
                             align_bert, tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForUnlabeledData(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=False,
    )
    src_dataset = Dataset.from_dict(src)
    src_dataset = src_dataset.map(
        partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=src_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    trg_dataset = Dataset.from_dict(trg)
    trg_dataset = trg_dataset.map(
        partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=trg_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )

    output = align_bert.mine_intersection_word_pairs(
        src_dataset, trg_dataset, threshold, collator, k=k, src_batch_size=1,
        trg_batch_size=1)
    output_idxs = [item[:4] for item in output]

    assert output_idxs == expected
    assert all([item[4] is not None for item in output])


@pytest.mark.parametrize('src,trg,threshold,k', [
    (
        {
            'text': [
                'I like beer .',
                'Du learnst Deutsch .',
                'I like beer .',
                'ThisTokenIsSplit to BPEs'
            ],
            'language': [
                'en',
                'de',
                'en',
                'en',
            ],
        },
        {
            'text': [
                'I like beer .',
                'Du learnst Deutsch .',
                'ThisTokenIsSplit to BPEs'
            ],
            'language': [
                'en',
                'en',
                'en',
            ],
        },
        0.0,
        1,
    ),
])
def test_mining_data_loader(src, trg, threshold, k,
                            align_bert, tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
    )
    src_dataset = Dataset.from_dict(src)
    tokenized_src_dataset = src_dataset.map(
        partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=src_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    trg_dataset = Dataset.from_dict(trg)
    tokenized_trg_dataset = trg_dataset.map(
        partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=trg_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )

    data_loader = cu.MiningDataLoader(
        src_dataset,
        trg_dataset,
        tokenized_src_dataset,
        tokenized_trg_dataset,
        1,
        align_bert,
        tokenizer_bert_multilingual_cased,
        threshold,
        collator,
        k,
        1
    )

    for item in data_loader:
        pass


@pytest.mark.parametrize('src_train,trg_train,eval', [
    (
        {
            'text': [
                'I like beer .',
                'I like beer .',
            ],
            'language': [
                'en',
                'en',
            ],
        },
        {
            'text': [
                'I like beer .',
                'Ich mag Bier .'
            ],
            'language': [
                'en',
                'en',
            ],
        },
        {
            'source': [
                # needs to have at least k=10 tokens
                'I like water . 5 6 7 8 9 10',
            ],
            'target': [
                'I like water . 5 6 7 8 9 10',
            ],
            'alignment': [
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                 (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)],
            ],
            'source_language': [
                'en',
            ],
            'target_language': [
                'en',
            ],
        },
    ),
])
def test_unsupervised_trainer(src_train, trg_train, eval,
                              bert_base, align_bert,
                              tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
    )
    src_train_dataset = Dataset.from_dict(src_train)
    trg_train_dataset = Dataset.from_dict(trg_train)
    train_dataset = cu.MultiDataset({
        'src': src_train_dataset,
        'trg': trg_train_dataset,
    })
    tokenized_train_dataset = cu.MultiDataset({
        k: v.map(
            partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased, 512),
            batched=True,
            num_proc=1,
            remove_columns=v.column_names,
            load_from_cache_file=False,
            desc=f"Running tokenizer on every text in dataset: {k}",
        )
        for k, v in train_dataset.datasets.items()
    })
    lang_pairs = [('src', 'trg'), ('trg', 'src')]
    eval_dataset = Dataset.from_dict(eval)
    eval_dataset = eval_dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    eval_dataset = cu.SizedMultiDataset({'test': eval_dataset})
    trainer = cm.UnsupervisedTrainer(
        model=align_bert,
        args=rc.MyTrainingArguments(
            detailed_logging=True,
            output_dir='/tmp',
            logging_dir='/tmp',
            logging_steps=1,
            save_steps=1000,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            evaluation_strategy='steps',
            eval_steps=2,
            mining_threshold=0.5,
            mining_k=1,
            mining_src_batch_size=1,
            mining_trg_batch_size=1,
            max_steps=10,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer_bert_multilingual_cased,
        data_collator=collator,
        bert_base=bert_base,
        include_clssep=rc.ModelArguments().include_clssep,
        language_pairs=lang_pairs,
        tokenized_train_dataset=tokenized_train_dataset,
    )
    trainer.train()
    trainer.evaluate()


@pytest.mark.parametrize('data,num', [
    (
        {
            'text': [
                'I like beer . 1',
                'I like beer . 2',
                'I like beer . 3',
                'I like beer . 4',
                'I like beer . 5',
                'I like beer . 6',
                'I like beer . 7',
            ],
            'language': [
                'en',
                'en',
                'en',
                'en',
                'en',
                'en',
                'en',
            ],
        },
        3,
    ),
    (
        {
            'text': [
                'I like beer . 1',
                'I like beer . 2',
                'I like beer . 3',
                'I like beer . 4',
                'I like beer . 5',
                'I like beer . 6',
                'I like beer . 7',
            ],
            'language': [
                'en',
                'en',
                'en',
                'en',
                'en',
                'en',
                'en',
            ],
        },
        7,
    ),
])
def test_data_sampler(data, num):
    dataset = Dataset.from_dict(data)
    sampled_dataset, sampled_dataset2 = cu.MiningDataLoader.sample_dataset(
        [dataset, dataset],
        num,
    )

    data_len = len(dataset)
    if num >= data_len:
        assert sampled_dataset == dataset
        assert sampled_dataset2 == dataset
    else:
        assert len(sampled_dataset) == num
        assert len(sampled_dataset2) == num
        assert [t for t in sampled_dataset['text']] == [t for t in sampled_dataset2['text']]


@pytest.mark.parametrize('train', [
    (
        {
            'source': [
                'I like beer .',
                'I like beer .',
            ],
            'target': [
                'I like beer .',
                'Ich mag Bier .'
            ],
            'alignment': [
                [(0, 0), (1, 1), (2, 2), (3, 3)],
                [(0, 0), (1, 1), (2, 2), (3, 3)],
            ],
            'source_language': [
                'de',
                'de',
            ],
            'target_language': [
                'en',
                'en',
            ]
        }
    ),
])
def test_freezed_linear(train,
                        bert_base, linear_bert,
                        tokenizer_bert_multilingual_cased):
    clone_model = copy.deepcopy(linear_bert)
    collator = cu.DataCollatorForAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=linear_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
    )
    train_dataset = Dataset.from_dict(train)
    train_dataset = train_dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased, 512),
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    train_dataset = cu.SizedMultiDataset({'test': train_dataset})
    trainer = cm.SupervisedTrainer(
        model=linear_bert,
        args=rc.MyTrainingArguments(
            detailed_logging=True,
            output_dir='/tmp',
            logging_steps=1,
            save_steps=1000,
            num_train_epochs=1,
            per_device_train_batch_size=1,
        ),
        train_dataset=train_dataset,
        tokenizer=tokenizer_bert_multilingual_cased,
        data_collator=collator,
        bert_base=bert_base,
        include_clssep=rc.ModelArguments().include_clssep,
    )
    trainer.train()

    linear_bert = linear_bert.to(clone_model.device)
    for param, clone_param in zip(
            linear_bert.bert.named_parameters(),
            clone_model.bert.named_parameters()
    ):
        assert torch.all(torch.eq(param[1], clone_param[1])), f'{param[0]} - {clone_param[0]}'

    for param, clone_param in zip(
        linear_bert._get_language_layer('de').named_parameters(),
        clone_model._get_language_layer('de').named_parameters(),
    ):
        assert not torch.all(torch.eq(param[1], clone_param[1])), f'{param[0]} - {clone_param[0]}'


@pytest.mark.parametrize('raw,parsed', [
    (
        {
            'text': '\n'.join([
                ' ||| '.join([
                    'TRAIN I like beer .',
                    'TRAIN I like beer .',
                ]),
                ' ||| '.join([
                    'TRAIN I like beer .',
                    'TRAIN I like beer .',
                ]),
                ' ||| '.join([
                    'DEV I like beer .',
                    'DEV I like beer .',
                ]),
                ' ||| '.join([
                    'TEST I like beer .',
                    'TEST I like beer .',
                ]),
                ' ||| '.join([
                    'TEST I like beer .',
                    'TEST I like beer .',
                ]),
            ]),
            'align': '\n'.join([
                '0-0 1-1 2-2 3-3',
                '0-0 1-1 2-2 3-3',
                '0-0 1-1 2-2 3-3',
                '0-0 1-1 2-2 3-3',
                '0-0 1-1 2-2 3-3',
            ]),
            'src_lang': 'en',
            'trg_lang': 'en',
        },
        DatasetDict({
            'train': Dataset.from_dict({
                'source': [
                    'TRAIN I like beer .',
                    'TRAIN I like beer .',
                ],
                'target': [
                    'TRAIN I like beer .',
                    'TRAIN I like beer .',
                ],
                'alignment': [
                    [(0, 0), (1, 1), (2, 2), (3, 3)],
                    [(0, 0), (1, 1), (2, 2), (3, 3)],
                ],
                'source_language': [
                    'en',
                    'en',
                ],
                'target_language': [
                    'en',
                    'en',
                ]

            }),
            'dev': Dataset.from_dict({
                'source': [
                    'DEV I like beer .',
                ],
                'target': [
                    'DEV I like beer .',
                ],
                'alignment': [
                    [(0, 0), (1, 1), (2, 2), (3, 3)],
                ],
                'source_language': [
                    'en',
                ],
                'target_language': [
                    'en',
                ]

            }),
            'test': Dataset.from_dict({
                'source': [
                    'TEST I like beer .',
                    'TEST I like beer .',
                ],
                'target': [
                    'TEST I like beer .',
                    'TEST I like beer .',
                ],
                'alignment': [
                    [(0, 0), (1, 1), (2, 2), (3, 3)],
                    [(0, 0), (1, 1), (2, 2), (3, 3)],
                ],
                'source_language': [
                    'en',
                    'en',
                ],
                'target_language': [
                    'en',
                    'en',
                ]

            }),
        }),
    ),
])
def test_load_parallel_from_file(raw, parsed):
    try:
        tmps = tempfile.mkstemp(text=True)[1]
        with open(tmps, 'w') as fout:
            fout.write(raw['text'])

        tmpa = tempfile.mkstemp(text=True)[1]
        with open(tmpa, 'w') as fout:
            fout.write(raw['align'])

        parsed_raw = cu.load_parallel_data_from_file(
            tmps,
            tmpa,
            raw['src_lang'],
            raw['trg_lang'],
            [('train', 2), ('dev', 1), ('test', -1)],
            load_from_cache_file=False,
        )

        for split in ['train', 'dev', 'test']:
            assert parsed_raw[split]['source'] == parsed[split]['source']
            assert parsed_raw[split]['target'] == parsed[split]['target']
            assert parsed_raw[split]['alignment'] == parsed[split]['alignment']
            assert parsed_raw[split]['source_language'] == parsed[split]['source_language']
            assert parsed_raw[split]['target_language'] == parsed[split]['target_language']
    finally:
        if os.path.exists(tmps):
            os.remove(tmps)
        if os.path.exists(tmpa):
            os.remove(tmpa)


@pytest.mark.parametrize('raw,parsed', [
    (
        {
            'text': '\n'.join([
                'This is a dummy sentence',
                'This is a dummy sentence',
                'This is a dummy sentence',
                'This is a dummy sentence',
                'Der ist ein Apfel',  # too short
                'This is a dummy sentence',
                'Apfel ist der beste',
            ]),
            'lang': 'en',
        },
        DatasetDict({
            'train': Dataset.from_dict({
                'text': [
                    'This is a dummy sentence',
                    'This is a dummy sentence',
                    'This is a dummy sentence',
                    'This is a dummy sentence',
                    'This is a dummy sentence',
                ],
                'language': [
                    'en',
                    'en',
                    'en',
                    'en',
                    'en',
                ],
            }),
        }),
    ),
])
def test_load_text_from_file(raw, parsed):
    try:
        tmpf = tempfile.mkstemp(text=True)[1]
        with open(tmpf, 'w') as fout:
            fout.write(raw['text'])

        parsed_raw = cu.load_text_data_from_file(
            tmpf,
            raw['lang'],
            load_from_cache_file=False,
        )

        assert parsed_raw['train']['text'] == parsed['train']['text']
        assert parsed_raw['train']['language'] == parsed['train']['language']
    finally:
        if os.path.exists(tmpf):
            os.remove(tmpf)
