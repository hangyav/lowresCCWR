import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from transformers import (
    BertPreTrainedModel,
    BertModel,
)
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


class BertForCaoAlign(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.bert_base = None
        self.subword_to_token_strategy = SubwordToTokenStrategyLast()
        self.init_weights()

    def init_base_model(self, *args, **kwargs):
        # TODO this is not too nice
        self.bert_base = BertModel.from_pretrained(*args, **kwargs)
        for param in self.bert_base.parameters():
            param.requires_grad = False

    def forward(
        self,
        src_input_ids,
        src_word_ids_lst,
        src_special_word_masks,
        trg_input_ids,
        trg_word_ids_lst,
        trg_special_word_masks,
        src_attention_masks,
        trg_attention_masks,
        alignment,
        include_clssep=True,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None \
                else self.config.use_return_dict

        for param in self.bert_base.parameters():
            # just to check if BERT.from_pretrained changes this
            assert not param.requires_grad

        src_features = self._process_sentences(
                src_input_ids,
                src_attention_masks,
                src_word_ids_lst,
                src_special_word_masks,
                include_clssep,
                self.bert,
        )
        trg_features = self._process_sentences(
                trg_input_ids,
                trg_attention_masks,
                trg_word_ids_lst,
                trg_special_word_masks,
                include_clssep,
                self.bert,
        )
        trg_features_base = self._process_sentences(
                trg_input_ids,
                trg_attention_masks,
                trg_word_ids_lst,
                trg_special_word_masks,
                include_clssep,
                self.bert_base,
        )

        src_idxs, trg_idxs = self._get_ligned_indices(
                alignment,
                include_clssep,
                src_special_word_masks,
                trg_special_word_masks,
                src_word_ids_lst,
                trg_word_ids_lst,
        )

        alignment_loss = F.mse_loss(
                src_features[src_idxs[:, 0], src_idxs[:, 1]],
                trg_features[trg_idxs[:, 0], trg_idxs[:, 1]]
        )
        regularization_loss = F.mse_loss(
            trg_features.reshape(-1, trg_features.shape[-1]),
            trg_features_base.reshape(-1, trg_features_base.shape[-1]),
        )
        total_loss = alignment_loss + regularization_loss

        if not return_dict:
            raise NotImplementedError('What to return here?')

        return CaoAlignmentOutput(
                alignment_loss=alignment_loss,
                regularization_loss=regularization_loss,
                loss=total_loss,
        )

    def _process_sentences(self, input_ids, attention_masks, word_ids_lst,
                           special_word_masks, include_clssep, model):
        features = model(
            input_ids,
            attention_mask=attention_masks,
            output_hidden_states=False,
        )['last_hidden_state']

        features = self.subword_to_token_strategy.process(
                features,
                word_ids_lst,
                special_word_masks,
                include_clssep,
        )

        return features

    def _get_ligned_indices(self, alignment_lsts, include_clssep,
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
                src_idx = get_word_idx_of_subword(
                    (src_special_word_masks[i] == 1).nonzero()[1],
                    src_word_ids_lst[i]
                )
                trg_idx = get_word_idx_of_subword(
                    (trg_special_word_masks[i] == 1).nonzero()[1],
                    trg_word_ids_lst[i]
                )

                src_res.append([i, src_idx])
                trg_res.append([i, trg_idx])

        src_res = torch.tensor(src_res, dtype=int).to(self.device)
        trg_res = torch.tensor(trg_res, dtype=int).to(self.device)

        return src_res, trg_res


def get_word_idx_of_subword(position, word_ids_lst):
    for i, lst in enumerate(word_ids_lst):
        if position in lst:
            return i
    return -1


@dataclass
class CaoAlignmentOutput(ModelOutput):

    alignment_loss: Optional[torch.FloatTensor] = None
    regularization_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    #  logits: torch.FloatTensor = None
    #  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    #  attentions: Optional[Tuple[torch.FloatTensor]] = None


class SubwordToTokenStrategyBase():

    def process(self, features, word_ids_lsts, special_word_masks,
                include_clssep):
        length = max([len(lst) for lst in word_ids_lsts])
        res = torch.zeros(
                (features.shape[0], length, features.shape[2]),
                dtype=features.dtype
        )
        self._process(res, features, word_ids_lsts, special_word_masks,
                      include_clssep)
        return res

    def _process(self, output, features, word_ids_lsts, special_word_masks,
                 include_clssep):
        raise NotImplementedError


class SubwordToTokenStrategyLast(SubwordToTokenStrategyBase):

    def _process(self, output, features, word_ids_lsts, special_word_masks,
                 include_clssep):
        for i, (feature, word_ids_lst, special_word_mask) \
                in enumerate(zip(features, word_ids_lsts, special_word_masks)):
            ids = [
                    id_lst[-1]
                    for id_lst, mask in zip(word_ids_lst, special_word_mask)
                    if (include_clssep and (mask == 1 or mask == 0)) or (not include_clssep and mask == 0)
            ]
            output[i, :len(ids)] = features[i, ids]


class SubwordToTokenStrategyAvg(SubwordToTokenStrategyBase):

    def _process(self, output, features, word_ids_lsts, special_word_masks,
                 include_clssep):
        raise NotImplementedError
