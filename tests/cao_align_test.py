import pytest
import torch
from functools import partial
import run_cao as rc
from cao_align import cao_model as cm
from cao_align import utils as cu
from transformers import AutoTokenizer, BertModel
from datasets import Dataset


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
    return cm.BertForCaoAlign.from_pretrained('bert-base-multilingual-cased')


@pytest.fixture
def align_mlm_bert():
    return cm.BertForCaoAlignMLM.from_pretrained(
        'bert-base-multilingual-cased',
        src_mlm_weight=0.01,
        trg_mlm_weight=0.01,
    )


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
    output = cu.tokenize_function_per_input(tokenizer_bert_multilingual_cased,
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
            [[0.0]*5, [1.0]*5, [2.0]*5, [-1.0]*5, [-1.0]*5, [-1.0]*5],
            [[0.0]*5, [1.0]*5, [1.5]*5, [2.0]*5, [2.5]*5, [3.0]*5],
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
    output = cu.DataCollatorForCaoAlignment.get_aligned_indices(
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
    collator = cu.DataCollatorForCaoAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
    )
    dataset = Dataset.from_dict(examples)
    dataset = dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased),
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
        },
    ),
])
def test_trainer(train, eval,
                 bert_base, align_bert, tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForCaoAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
    )
    train_dataset = Dataset.from_dict(train)
    train_dataset = train_dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased),
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    train_dataset = cu.MultiDataset({'test': train_dataset})
    eval_dataset = Dataset.from_dict(eval)
    eval_dataset = eval_dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased),
        batched=True,
        num_proc=1,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    eval_dataset = cu.MultiDataset({'test': eval_dataset})
    trainer = cm.CaoTrainer(
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
        },
    ),
])
def test_trainer_mlm(train, eval,
                     bert_base, align_mlm_bert,
                     tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForCaoMLMAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_mlm_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
        mlm_probability=0.9,
        pad_to_multiple_of_8=False,
    )
    train_dataset = Dataset.from_dict(train)
    train_dataset = train_dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased),
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    train_dataset = cu.MultiDataset({'test': train_dataset})
    eval_dataset = Dataset.from_dict(eval)
    eval_dataset = eval_dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased),
        batched=True,
        num_proc=1,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    eval_dataset = cu.MultiDataset({'test': eval_dataset})
    trainer = cm.CaoTrainer(
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
])
def test_mlm_pipeline(examples, expected, equals,
                      bert_base, align_mlm_bert, tokenizer_bert_multilingual_cased):
    collator = cu.DataCollatorForCaoMLMAlignment(
        tokenizer=tokenizer_bert_multilingual_cased,
        max_length=align_mlm_bert.bert.embeddings.position_embeddings.num_embeddings,
        include_clssep=True,
        mlm_probability=1.0,
        pad_to_multiple_of_8=False,
    )
    dataset = Dataset.from_dict(examples)
    dataset = dataset.map(
        partial(cu.tokenize_function_for_parallel, tokenizer_bert_multilingual_cased),
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

    output = align_mlm_bert(return_dict=True, bert_base=bert_base, **batch)

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
        },
        {
            'text': [
                'I like beer .',
                'Du learnst Deutsch .',
                'ThisTokenIsSplit to BPEs'
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
        partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased),
        batched=True,
        num_proc=1,
        remove_columns=src_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )
    trg_dataset = Dataset.from_dict(trg)
    trg_dataset = trg_dataset.map(
        partial(cu.tokenize_function_for_unlabeled, tokenizer_bert_multilingual_cased),
        batched=True,
        num_proc=1,
        remove_columns=trg_dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on every text in dataset",
    )

    output = align_bert.mine_word_pairs(src_dataset, trg_dataset, threshold,
                                        collator, k=k, batch_size=1)
    output_idxs = [item[:4] for item in output]

    assert output_idxs == expected
    assert all([item[4] is not None for item in output])
