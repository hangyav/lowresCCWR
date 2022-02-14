import os
import logging
from collections.abc import Sized
from typing import Dict
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling

from datasets import Dataset

from mining_extractor import get_color, reset_color

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForUnlabeledData:

    tokenizer: PreTrainedTokenizerBase
    max_length: int
    include_clssep: bool

    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        max_len = max([len(item['input_ids']) for item in examples])

        input_ids = self._handle_input_ids(
                examples,
                'input_ids',
                max_len,
                self.tokenizer.pad_token_id,
        )

        special_word_masks = self._handle_special_word_masks(
                examples,
                'special_word_masks',
                input_ids,
        )

        attention_masks = (special_word_masks != 2).type(torch.IntTensor)
        word_ids_lst = self._to_list(examples, 'word_ids_lst')
        lang_lst = self._to_list(examples, 'language')

        return {
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'word_ids_lst': word_ids_lst,
            'special_word_masks': special_word_masks,
            'language': lang_lst,
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
            special_word_masks[i, :len(example[col])] = torch.tensor(
                example[col],
                dtype=int
            )

        return special_word_masks

    def _to_list(self, examples, col):
        return [example[col] for example in examples]

    def get_eval(self):
        return self


class DataCollatorForCaoAlignment(DataCollatorForUnlabeledData):

    def __init__(self, tokenizer, max_length, include_clssep):
        super().__init__(tokenizer, max_length, include_clssep)

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
        src_lang_lst = self._to_list(examples, 'src_language')
        trg_lang_lst = self._to_list(examples, 'trg_language')

        src_idxs, trg_idxs = DataCollatorForCaoAlignment.get_aligned_indices(
                alignment,
                self.include_clssep,
                src_special_word_masks,
                trg_special_word_masks,
                src_word_ids_lst,
                trg_word_ids_lst,
        )

        return {
            'src_input_ids': src_input_ids,
            'trg_input_ids': trg_input_ids,
            'src_attention_masks': src_attention_masks,
            'trg_attention_masks': trg_attention_masks,
            'src_word_ids_lst': src_word_ids_lst,
            'trg_word_ids_lst': trg_word_ids_lst,
            'src_special_word_masks': src_special_word_masks,
            'trg_special_word_masks': trg_special_word_masks,
            'src_alignments': src_idxs,
            'trg_alignments': trg_idxs,
            'src_language': src_lang_lst,
            'trg_language': trg_lang_lst,
        }

    @staticmethod
    def get_aligned_indices(alignment_lsts, include_clssep,
                            src_special_word_masks=None,
                            trg_special_word_masks=None,
                            src_word_ids_lst=None,
                            trg_word_ids_lst=None):
        assert not include_clssep or \
            (
                src_special_word_masks is not None and
                trg_special_word_masks is not None and
                src_word_ids_lst is not None and
                trg_word_ids_lst is not None
            )

        src_res = list()
        trg_res = list()
        offset = 0
        if include_clssep:
            # +1 because of added [CLS]
            offset = 1

        for i, alignment_lst in enumerate(alignment_lsts):
            for alignment in alignment_lst:
                src_res.append([i, alignment[0] + offset])
                trg_res.append([i, alignment[1] + offset])

            if include_clssep:
                # [CLS]
                src_res.append([i, 0])
                trg_res.append([i, 0])
                # [SEP]
                src_idx = DataCollatorForCaoAlignment.get_word_idx_of_subword(
                    (src_special_word_masks[i] == 1).nonzero()[1],
                    src_word_ids_lst[i]
                )
                trg_idx = DataCollatorForCaoAlignment.get_word_idx_of_subword(
                    (trg_special_word_masks[i] == 1).nonzero()[1],
                    trg_word_ids_lst[i]
                )

                src_res.append([i, src_idx])
                trg_res.append([i, trg_idx])

        src_res = torch.tensor(src_res, dtype=int)
        trg_res = torch.tensor(trg_res, dtype=int)

        return src_res, trg_res

    @staticmethod
    def get_word_idx_of_subword(position, word_ids_lst):
        for i, lst in enumerate(word_ids_lst):
            if position in lst:
                return i
        return -1


class DataCollatorForCaoMLMAlignment(DataCollatorForCaoAlignment):

    eval_mode: bool

    def __init__(self, tokenizer, max_length, include_clssep,
                 mlm_probability=0.15, pad_to_multiple_of_8=False,
                 eval_mode=False):
        super().__init__(tokenizer, max_length, include_clssep)
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
        self.eval_mode = eval_mode

    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        output = super().__call__(examples)

        if self.eval_mode:
            return output

        for prefix in ['src', 'trg']:
            input_ids = output[f'{prefix}_input_ids']
            special_token_masks = output[f'{prefix}_special_word_masks']
            special_token_masks = (special_token_masks >
                                   0).type(torch.IntTensor)

            masked_inputs, labels = self.mlm_collator.mask_tokens(
                input_ids.clone(), special_token_masks)

            output[f'{prefix}_mlm_input_ids'] = masked_inputs
            output[f'{prefix}_mlm_labels'] = labels

        return output

    def get_eval(self):
        return DataCollatorForCaoMLMAlignment(
            self.tokenizer,
            self.max_length,
            self.include_clssep,
            self.mlm_collator.mlm_probability,
            self.mlm_collator.pad_to_multiple_of,
            True,
        )


class DataCollatorForNER(DataCollatorForUnlabeledData):

    def __init__(self, tokenizer, max_length, label_pad_token_id=-100):
        super().__init__(tokenizer, max_length, True)
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        res = super().__call__(examples)

        max_len = max([len(item) for item in res['word_ids_lst']])
        labels = torch.zeros((len(examples), max_len), dtype=int)
        labels = labels.fill_(self.label_pad_token_id)

        for i, example in enumerate(examples):
            labels[i, :len(example['labels'])] = torch.tensor(
                example['labels'],
                dtype=int
            )
        res['labels'] = labels

        return res


class MultiDataset():

    def __init__(self, datasets):
        self.datasets = datasets


class SizedMultiDataset(MultiDataset, Sized):

    def __init__(self, datasets):
        MultiDataset.__init__(self, datasets)

    def __len__(self):
        return sum([len(d) for d in self.datasets.values()])


class MultiDataLoader():

    def __init__(self, dataset, data_loaders):
        self.dataset = dataset  # needed for some checks in transformer Trainer class
        self.data_loaders = data_loaders

    def __iter__(self):
        iters = [d.__iter__() for d in self.data_loaders]
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


class SizedMultiDataLoader(MultiDataLoader, Sized):

    def __init__(self, dataset, data_loaders, batch_size):
        MultiDataLoader.__init__(self, dataset, data_loaders)
        self.batch_size = batch_size

    def __len__(self):
        return round(len(self.dataset) / self.batch_size + 0.5)


class MiningDataLoader():

    mine_fns = {
        'forward': 'mine_word_pairs',
        'intersection': 'mine_intersection_word_pairs',
    }

    def __init__(self, src_dataset, trg_dataset, batch_size, model, tokenizer,
                 threshold, parallel_collator, k=1, mine_batch_size=None,
                 dataloader_num_workers=0, dataloader_pin_memory=True,
                 sample_for_mining=None, threshold_max=100, log_dir=None,
                 mining_method='intersection', num_dataset_iterations=1,
                 max_seq_length=None, use_data_cache=True):
        self.dataset = (src_dataset, trg_dataset)
        self.batch_size = batch_size
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.parallel_collator = parallel_collator
        self.k = k
        self.mine_batch_size = mine_batch_size if mine_batch_size else batch_size
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_num_workers = dataloader_num_workers
        self.sample_for_mining = sample_for_mining
        self.threshold_max = threshold_max
        self.log_dir = log_dir
        self._num_minings = 0
        self.num_dataset_iterations = num_dataset_iterations
        assert mining_method in self.mine_fns, f'Method {mining_method} not supported'
        self.mining_method = mining_method
        self.max_seq_length = max_seq_length
        self.use_data_cache = use_data_cache

        self._unlabeled_collator = DataCollatorForUnlabeledData(
            tokenizer=tokenizer,
            max_length=model.bert.embeddings.position_embeddings.num_embeddings,
            include_clssep=False,
        )

        self._tokenized_src_dataset = self._tokenize_dataset(
            src_dataset,
            #  tokenize_function_for_unlabeled,
            self._tokenizer_fn_unlabeled,
        )
        #  assert trg_dataset['language'][0] == 'en', ('Not implemented: In case'
        #                                              + ' of En as target'
        #                                              + ' language this needs to'
        #                                              + ' be handled in the'
        #                                              + ' models!!!')
        self._tokenized_trg_dataset = self._tokenize_dataset(
            trg_dataset,
            #  tokenize_function_for_unlabeled,
            self._tokenizer_fn_unlabeled,
        )
        self._tmp_data_loader = None

    def _tokenizer_fn_unlabeled(self, examples):
        return tokenize_function_for_unlabeled(
            self.tokenizer,
            self.max_seq_length,
            examples,
        )

    def _tokenizer_fn_parallel(self, examples):
        return tokenize_function_for_parallel(
            self.tokenizer,
            self.max_seq_length,
            examples,
        )

    def _tokenize_dataset(self, dataset, tok_fv):
        return dataset.map(
            #  partial(tok_fv, self.tokenizer, self.max_seq_length),
            tok_fv,
            batched=True,
            num_proc=self.dataloader_num_workers if self.dataloader_num_workers else 1,
            remove_columns=dataset.column_names,
            load_from_cache_file=self.use_data_cache,
        )

    @staticmethod
    def sample_dataset(datasets, num):
        assert len({len(d) for d in datasets}) == 1
        if num >= len(datasets[0]):
            return datasets

        indices = np.random.choice(
            range(len(datasets[0])),
            size=num,
            replace=False,
        )
        return [
            Dataset.from_dict(dataset[indices])
            for dataset in datasets
        ]

    def _mine(self):
        model_training = self.model.training
        self.model.eval()

        tokenized_src_dataset = self._tokenized_src_dataset
        tokenized_trg_dataset = self._tokenized_trg_dataset
        src_dataset = self.dataset[0]
        trg_dataset = self.dataset[1]
        if self.sample_for_mining:
            tokenized_src_dataset, src_dataset = self.sample_dataset(
                [tokenized_src_dataset, src_dataset],
                self.sample_for_mining,
            )
            tokenized_trg_dataset, trg_dataset = self.sample_dataset(
                [tokenized_trg_dataset, trg_dataset],
                self.sample_for_mining,
            )

        with torch.no_grad():
            mine_fn = getattr(self.model, self.mine_fns[self.mining_method])
            mining = mine_fn(
                tokenized_src_dataset,
                tokenized_trg_dataset,
                self.threshold,
                self._unlabeled_collator,
                k=self.k,
                batch_size=self.mine_batch_size,
                threshold_max=self.threshold_max,
            )

        if model_training:
            self.model.train()

        tmp_lst = None
        last_sent_pair = None
        res = dict()
        for item in mining:
            curr_sent_pair = (item[0], item[2])

            if last_sent_pair != curr_sent_pair:
                self._process(tmp_lst, src_dataset, trg_dataset, res)
                tmp_lst = list()
                last_sent_pair = curr_sent_pair

            tmp_lst.append(item)

        self._process(tmp_lst, src_dataset, trg_dataset, res)
        self._log_if_needed(res)
        res.pop('score')

        return Dataset.from_dict(res)

    def _process(self, align_lst, src_dataset, trg_dataset, out_dict):
        if align_lst is None or len(align_lst) == 0:
            return

        src_sent_id = align_lst[0][0]
        trg_sent_id = align_lst[0][2]
        src_sent = src_dataset['text'][src_sent_id]
        trg_sent = trg_dataset['text'][trg_sent_id]
        src_len = len(src_sent.split())
        trg_len = len(trg_sent.split())
        src_lang = src_dataset['language'][src_sent_id]
        trg_lang = trg_dataset['language'][trg_sent_id]
        alignments = [
            [align[1], align[3]]
            for align in align_lst
            if align[1] < src_len and align[3] < trg_len  # eliminate padding alignments
        ]
        scores = [align[4] for align in align_lst]

        out_dict.setdefault('source', list()).append(src_sent)
        out_dict.setdefault('target', list()).append(trg_sent)
        out_dict.setdefault('source_language', list()).append(src_lang)
        out_dict.setdefault('target_language', list()).append(trg_lang)
        out_dict.setdefault('alignment', list()).append(alignments)
        out_dict.setdefault('score', list()).append(scores)

    def _log_if_needed(self, dataset):
        if self.log_dir is not None:
            with open(os.path.join(self.log_dir, f'minings.{self._num_minings}.log'), 'w') as fout:
                for i in range(len(dataset['source'])):
                    s1 = dataset['source'][i].split()
                    s2 = dataset['target'][i].split()

                    for idx, (align, score) in enumerate(zip(
                        dataset['alignment'][i],
                        dataset['score'][i],
                    )):
                        try:
                            print(f'{get_color(idx)}{align[0]}:{s1[align[0]]}'
                                  + f' - {align[1]}:{s2[align[1]]}'
                                  + f' - {score}{reset_color}', file=fout)
                            s1[align[0]] = f'{get_color(idx)}{s1[align[0]]}{reset_color}'
                            s2[align[1]] = f'{get_color(idx)}{s2[align[1]]}{reset_color}'
                        except Exception as e:
                            print(s1)
                            print(s2)
                            print(align)
                            print(score)
                            raise e

                    print(' '.join(s1), file=fout)
                    print(' '.join(s2), file=fout)
                    print('', file=fout)
            self._num_minings += 1

    def _get_data_loader(self):
        if self._tmp_data_loader is not None and self._tmp_data_loader[1] < 1:
            self._tmp_data_loader = None

        if self._tmp_data_loader is None:
            dataset = self._mine()
            dataset = self._tokenize_dataset(
                dataset,
                #  tokenize_function_for_parallel
                self._tokenizer_fn_parallel,
            )

            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=self.parallel_collator,
                num_workers=self.dataloader_num_workers,
                pin_memory=self.dataloader_pin_memory,
            )
            self._tmp_data_loader = (data_loader, self.num_dataset_iterations)

        self._tmp_data_loader = (
            self._tmp_data_loader[0],
            self._tmp_data_loader[1] - 1,
        )
        return self._tmp_data_loader[0]

    def __iter__(self):
        return self._get_data_loader().__iter__()


def cat_tensors_with_padding(a, b, value=0):
    diff = a.shape[1] - b.shape[1]
    if diff > 0:
        shape = list(b.shape)
        shape[1] = diff
        tmp = torch.ones(*shape, device=b.device) * value
        b = torch.cat((b, tmp), 1)
    elif diff < 0:
        shape = list(a.shape)
        shape[1] = abs(diff)
        tmp = torch.ones(*shape, device=a.device) * value
        a = torch.cat((a, tmp), 1)

    return torch.cat((a, b), 0)


def detokenize_sentence(input_ids, tokenizer):
    toks = tokenizer.convert_ids_to_tokens(input_ids)
    toks = ' '.join(toks)
    toks = toks.replace(' ##', '').split(' ')

    return toks


def detokenize(input_ids, tokenizer):
    res = list()
    for ids in input_ids:
        res.append(detokenize_sentence(ids, tokenizer))
    return res


def normalize_matrix(mat, dim=-1):
    norm = mat.norm(dim=-1, keepdim=True)
    norm[norm < 1e-5] = 1.0
    return mat / norm


def sentence_batch_cosine_similarity(sentence, batch):
    """
    Embeddings of a sentence: num_sent_words x emb_dim
    Embeddings in a batch: num_batch_sentences x num_batch_words x emb_dim

    output: num_batch_sentences x num_sent_words x num_batch_words
    """
    sentence = normalize_matrix(sentence)
    batch = normalize_matrix(batch)

    res = sentence.matmul(batch.transpose(1, 2))
    # Zero vectors (PAD) give nan values, set them to 0.0
    res = res.nan_to_num()
    return res


def batch_batch_cosine_similarity(batch1, batch2):
    """
    Embeddings in a batch1: num_batch_sentences1 x num_batch_words1 x emb_dim
    Embeddings in a batch2: num_batch_sentences2 x num_batch_words2 x emb_dim

    output: num_batch_sentences1 x num_batch_sentences2 x num_batch_words1
            x num_batch_words2
    """
    batch1 = normalize_matrix(batch1)
    batch2 = normalize_matrix(batch2)

    res = batch1.unsqueeze(0).transpose(0, 1).matmul(batch2.transpose(1, 2))
    # Zero vectors (PAD) give nan values, set them to 0.0
    res = res.nan_to_num()
    return res


def tokenize_function_per_input(tokenizer, max_seq_length, examples):
    # this contains the subword token ids
    input_ids_lst = list()
    # contains lists of input_ids indices to show which subword tokens
    # belong to which words
    word_ids_lst_lst = list()
    # contains 1 if special token like SEP or CLS and o otherwise
    special_word_masks_lst = list()

    for example in examples:
        input_ids = list()
        word_ids_lst = list()
        special_word_masks = list()

        input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.cls_token))
        word_ids_lst.append([0])
        special_word_masks.append(1)

        if type(example) == str:
            example = example.split()

        for word in example:
            tokens = tokenizer.tokenize(word)
            if len(tokens) == 0:
                # XXX why doesn't this happen automatically?
                tokens = [tokenizer.unk_token]
            ids = tokenizer.convert_tokens_to_ids(tokens)

            cur_len = len(input_ids)
            ids_len = len(ids)

            if max_seq_length is not None and cur_len + ids_len > max_seq_length - 1:  # -1 for [SEP] at the end
                break

            input_ids.extend(ids)

            word_ids_lst.append(list(range(cur_len, cur_len+ids_len)))

            special_word_masks.extend([0]*len(ids))

        word_ids_lst.append([len(input_ids)])
        input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))
        special_word_masks.append(1)

        input_ids_lst.append(input_ids)
        word_ids_lst_lst.append(word_ids_lst)
        special_word_masks_lst.append(special_word_masks)

    result = {
        'input_ids': input_ids_lst,
        'word_ids_lst': word_ids_lst_lst,
        'special_word_masks': special_word_masks_lst,
    }
    return result


def tokenize_function_for_parallel(tokenizer, max_seq_length, examples):
    src = tokenize_function_per_input(tokenizer, max_seq_length, examples['source'])
    trg = tokenize_function_per_input(tokenizer, max_seq_length, examples['target'])
    #  assert examples['target_language'][0] == 'en', ('Not implemented: In case'
    #                                                  + ' of En as target'
    #                                                  + ' language this needs to'
    #                                                  + ' be handled in the'
    #                                                  + ' models!!!')
    alignments = filter_alignment(
        src['word_ids_lst'],
        trg['word_ids_lst'],
        examples['alignment'],
    )
    return {
        'src_input_ids': src['input_ids'],
        'src_word_ids_lst': src['word_ids_lst'],
        'src_special_word_masks': src['special_word_masks'],
        'trg_input_ids': trg['input_ids'],
        'trg_word_ids_lst': trg['word_ids_lst'],
        'trg_special_word_masks': trg['special_word_masks'],
        #  'alignment': examples['alignment'],
        'alignment': alignments,
        'src_language': examples['source_language'],
        'trg_language': examples['target_language'],
    }


def filter_alignment(src_word_ids_lst, trg_word_ids_lst, alignemnts):
    res = list()
    for src_lst, trg_lst, alignment in zip(src_word_ids_lst, trg_word_ids_lst, alignemnts):
        # -2 to compensate [CLS] and [SEP]
        num_src_word = len(src_lst) - 2
        num_trg_word = len(trg_lst) - 2
        res.append([
            [src_a, trg_a]
            for src_a, trg_a in alignment
            if src_a < num_src_word and trg_a < num_trg_word
        ])

    return np.array(res)


def tokenize_function_for_unlabeled(tokenizer, max_seq_length, examples):
    res = tokenize_function_per_input(tokenizer, max_seq_length, examples['text'])
    res['language'] = examples['language']

    return res


def tokenizer_function_for_ner(tokenizer, text_column_name, label_column_name,
                               label_to_id, max_seq_length, examples):
    res = tokenize_function_per_input(
        tokenizer,
        max_seq_length,
        examples[text_column_name]
    )
    res['language'] = [examples['langs'][0][0]] * len(examples['langs'])

    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_id_lsts = res['word_ids_lst'][i]
        word_masks = res['special_word_masks'][i]
        label_ids = []
        word_pos = 0
        for word_id_lst in word_id_lsts:
            # Special tokens have a word mask of 1. We set the label
            # to -100 so they are automatically ignored in the loss
            # function.
            if len(word_id_lst) == 1 and word_masks[word_id_lst[0]] == 1:
                label_ids.append(-100)
            else:
                label_ids.append(label_to_id[label[word_pos]])
                word_pos += 1

        labels.append(label_ids)

    res["labels"] = labels

    return res


def save_embeddings(embeddings, output, word_order=None):
    with open(output, 'w') as fout:
        num = len(embeddings)
        dim = embeddings[next(embeddings.keys().__iter__())].shape[0]
        print(f'{num} {dim}', file=fout)

        if word_order is not None:
            words = [
                w
                for w in word_order
                if w in embeddings
            ]
        else:
            words = embeddings.keys()

        for w in words:
            print('{} {}'.format(
                    w,
                    ' '.join(['%.6g' % x for x in embeddings[w]]),
                ),
                file=fout,
            )
