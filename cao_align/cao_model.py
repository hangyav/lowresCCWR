import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Sized

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from transformers import (
    BertPreTrainedModel,
    BertModel,
    is_datasets_available,
)
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments, ParallelMode
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer, _is_torch_generator_available
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import (
        IterableDatasetShard,
        LengthGroupedSampler,
        DistributedLengthGroupedSampler,
        RandomSampler,
        DistributedSampler,
        DistributedSamplerWithLoop,
)

from cao_align.cao_data import MultiDataLoader

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


class CaoTrainer(Trainer):

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers
        )

    def get_train_dataloader(self):
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        datasets = self.train_dataset.values()
        datasets = [self._get_single_train_dataloader(d) for d in datasets]

        if len(datasets) == 1:
            return datasets[0]

        return MultiDataLoader(datasets)

    def _get_single_train_dataloader(self, train_dataset) -> DataLoader:
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.dataset.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_single_train_sampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_single_train_sampler(self, train_dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(train_dataset, Sized):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(train_dataset, Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    train_dataset,
                    self.args.train_batch_size,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(train_dataset, generator=generator)
                return RandomSampler(train_dataset)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
