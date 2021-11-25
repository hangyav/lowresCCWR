from collections.abc import Sized
import torch
from dataclasses import dataclass
from typing import Dict
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling


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

        return {
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'word_ids_lst': word_ids_lst,
            'special_word_masks': special_word_masks,
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


class MultiDataset(Sized):

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum([len(d) for d in self.datasets.values()])


class MultiDataLoader(Sized):

    def __init__(self, dataset, data_loaders, batch_size):
        self.data_loaders = data_loaders
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return round(len(self.dataset) / self.batch_size + 0.5)

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


def cat_tensors_with_padding(a, b, value=0):
    diff = a.shape[1] - b.shape[1]
    if diff > 0:
        shape = list(b.shape)
        shape[1] = diff
        tmp = torch.ones(*shape) * value
        b = torch.cat((b, tmp), 1)
    elif diff < 0:
        shape = list(a.shape)
        shape[1] = abs(diff)
        tmp = torch.ones(*shape) * value
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


def tokenize_function_per_input(tokenizer, examples):
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

        for word in example.split():
            ids = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(word)
            )
            assert len(ids) > 0
            cur_len = len(input_ids)
            ids_len = len(ids)
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


def tokenize_function_for_parallel(tokenizer, examples):
    src = tokenize_function_per_input(tokenizer, examples['source'])
    trg = tokenize_function_per_input(tokenizer, examples['target'])
    return {
        'src_input_ids': src['input_ids'],
        'src_word_ids_lst': src['word_ids_lst'],
        'src_special_word_masks': src['special_word_masks'],
        'trg_input_ids': trg['input_ids'],
        'trg_word_ids_lst': trg['word_ids_lst'],
        'trg_special_word_masks': trg['special_word_masks'],
        'alignment': examples['alignment'],
    }


def tokenize_function_for_unlabeled(tokenizer, examples):
    return tokenize_function_per_input(tokenizer, examples['text'])
