from collections.abc import Sized
import torch
from dataclasses import dataclass
from typing import Dict
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForCaoAlignment:

    tokenizer: PreTrainedTokenizerBase
    max_length: int
    include_clssep: bool

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

    def _handle_input_ids(self, examples, col, max_len, default_value):
        input_ids = torch.zeros((len(examples), max_len), dtype=int)
        input_ids = input_ids.fill_(default_value)

        for i, example in enumerate(examples):
            input_ids[i, :len(example[col])] = torch.tensor(example[col], dtype=int)

        return input_ids

    def _handle_special_word_masks(self, examples, col, input_ids):
        special_word_masks = torch.zeros_like(input_ids).fill_(2)

        for i, example in enumerate(examples):
            special_word_masks[i, :len(example[col])] = torch.tensor(example[col], dtype=int)

        return special_word_masks

    def _to_list(self, examples, col):
        return [example[col] for example in examples]

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

    def get_eval(self):
        return self


class DataCollatorForCaoMLMAlignment(DataCollatorForCaoAlignment):

    tokenizer: PreTrainedTokenizerBase
    max_length: int
    include_clssep: bool
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
