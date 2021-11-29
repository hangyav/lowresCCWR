import logging
from tqdm.auto import tqdm
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Sized

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from datasets import Dataset as HugDataset
from transformers import (
    BertPreTrainedModel,
    BertModel,
    is_datasets_available,
)
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.modeling_outputs import MaskedLMOutput
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

from cao_align.multilingual_alignment import hubness_CSLS, bestk_idx_CSLS
from cao_align.utils import (
    MultiDataLoader,
    cat_tensors_with_padding,
    detokenize,
    sentence_batch_cosine_similarity,
)

logger = logging.getLogger(__name__)


class BertForCaoAlign(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.subword_to_token_strategy = SubwordToTokenStrategyLast()
        self.init_weights()

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
        src_alignments,
        trg_alignments,
        include_clssep=True,
        return_dict=None,
        bert_base=None,
    ):
        return_dict = return_dict if return_dict is not None \
                else self.config.use_return_dict

        if not return_dict:
            raise NotImplementedError('What to return here?')

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

        if bert_base is not None:
            trg_features_base = self._process_sentences(
                    trg_input_ids,
                    trg_attention_masks,
                    trg_word_ids_lst,
                    trg_special_word_masks,
                    include_clssep,
                    bert_base,
            )

            alignment_loss = F.mse_loss(
                    src_features[src_alignments[:, 0], src_alignments[:, 1]],
                    trg_features_base[trg_alignments[:, 0], trg_alignments[:, 1]]
            )

            regularization_loss = F.mse_loss(
                trg_features.reshape(-1, trg_features.shape[-1]),
                trg_features_base.reshape(-1, trg_features_base.shape[-1]),
            )
        else:
            regularization_loss = torch.zeros(
                (1, 1), dtype=torch.float).to(self.bert.device)
            alignment_loss = torch.zeros(
                (1, 1), dtype=torch.float).to(self.bert.device)

        total_loss = alignment_loss + regularization_loss

        return CaoAlignmentOutput(
                alignment_loss=alignment_loss,
                regularization_loss=regularization_loss,
                loss=total_loss,
                src_hidden_states=src_features,
                trg_hidden_states=trg_features,
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

    def mine_word_pairs(self, src_data, trg_data, threshold, data_collator,
                        k=1, batch_size=16):
        """
        output: [(src_sentence_idx, src_word_idx, trg_sentence_idx, trg_word_idx)]
        """
        res = list()
        src_batch_offset = 0
        src_data_loader = DataLoader(
            src_data,
            batch_size=batch_size,
            collate_fn=data_collator,
        )
        for src_batch in tqdm(src_data_loader, desc='Mining'):
            src_batch_features = self._process_sentences(
                    src_batch['input_ids'].to(self.bert.device),
                    src_batch['attention_masks'].to(self.bert.device),
                    src_batch['word_ids_lst'],
                    src_batch['special_word_masks'].to(self.bert.device),
                    False,
                    self.bert,
            )
            batch_res = dict()
            trg_data_loader = DataLoader(
                trg_data,
                batch_size=batch_size,
                collate_fn=data_collator,
            )
            trg_batch_offset = 0
            for trg_batch in trg_data_loader:
                trg_batch_features = self._process_sentences(
                        trg_batch['input_ids'].to(self.bert.device),
                        trg_batch['attention_masks'].to(self.bert.device),
                        trg_batch['word_ids_lst'],
                        trg_batch['special_word_masks'].to(self.bert.device),
                        False,
                        self.bert,
                )

                for src_sent_idx, (src_sent_feats, src_batch_item) in enumerate(
                        zip(src_batch_features, src_batch)
                ):
                    # TODO can we eliminate the loop above?
                    similarities = sentence_batch_cosine_similarity(
                        src_sent_feats,
                        trg_batch_features,
                    )
                    similarities[similarities < threshold] = 0.0

                    #  zero out elements which belong to a src or trg token
                    #  which is a PAD
                    #  XXX below is buggy/not finished. But we don't need it
                    #  actually, they don't get mined since cos of zero vector
                    #  is low
                    #  src_non_pad_indices = src_sent_feats.sum(dim=-1) != 0.0
                    #  similarities[:, src_non_pad_indices, :] = 0.0
                    #  trg_non_pad_indices = trg_batch_features.sum(dim=-1) != 0.0
                    #  similarities[trg_non_pad_indices.unsqueeze(
                    #      1).repeat(1, similarities.shape[1], 1)] = 0.0

                    for index in similarities.nonzero():
                        # index[0]: trg_sent_idx
                        # index[1]: src_word_idx
                        # index[2]: trg_word_idx
                        batch_res.setdefault(
                            (src_sent_idx+src_batch_offset, index[1].item()),
                            list()
                        ).append((
                            index[0].item()+trg_batch_offset,
                            index[2].item(),
                            similarities[index[0], index[1], index[2]].item(),
                        ))

                trg_batch_offset += trg_data_loader.batch_size

            for key, v in batch_res.items():
                v = list(sorted(v, key=lambda x: x[2], reverse=True))[:k]
                for item in v:
                    res.append((key[0], key[1], item[0], item[1], item[2]))

            src_batch_offset += src_data_loader.batch_size

        return list(sorted(res, key=lambda x: (x[0], x[2], x[1], x[3])))

    def mine_intersection_word_pairs(self, src_data, trg_data, threshold,
                                     data_collator, k=1, batch_size=16):
        with tqdm(desc='Intersection mining', total=3) as pbar:
            res = list()
            forward = self.mine_word_pairs(src_data, trg_data, threshold,
                                           data_collator, k, batch_size)
            pbar.update(1)

            forward_dict = dict()
            for item in forward:
                forward_dict.setdefault(
                    item[0],
                    dict(),
                ).setdefault(
                    item[2],
                    dict(),
                )[(item[1], item[3])] = item[4]
            pbar.update(1)

            for src_idx, trg in tqdm(forward_dict.items()):
                trg_idxs = list(sorted(trg.keys()))
                src_sent = HugDataset.from_dict(src_data[[src_idx]])
                trg_sents = HugDataset.from_dict(trg_data[trg_idxs])
                backward = self.mine_word_pairs(
                    trg_sents,
                    src_sent,
                    threshold,
                    data_collator,
                    k,
                    batch_size
                )

                for item in backward:
                    mined_trg_idx = trg_idxs[item[0]]
                    reverse_mined_word_pair_idxs = (item[3], item[1])
                    if (mined_trg_idx in forward_dict[src_idx]
                            and reverse_mined_word_pair_idxs in forward_dict[src_idx][mined_trg_idx]):
                        res.append((
                            src_idx,
                            reverse_mined_word_pair_idxs[0],
                            mined_trg_idx,
                            reverse_mined_word_pair_idxs[1],
                            np.mean([forward_dict[src_idx][mined_trg_idx]
                                    [reverse_mined_word_pair_idxs], item[4]])
                        ))
            pbar.update(1)

            return list(sorted(res, key=lambda x: (x[0], x[2], x[1], x[3])))


class BertForCaoAlignMLM(BertForCaoAlign):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"predictions.decoder.bias",
    ]

    def __init__(self, config, src_mlm_weight, trg_mlm_weight):
        super().__init__(config)
        self.cls = BertOnlyMLMHead(config)
        self.src_mlm_weight = src_mlm_weight
        self.trg_mlm_weight = trg_mlm_weight
        assert self.src_mlm_weight != 0.0 or self.trg_mlm_weight != 0.0, \
            'Either source or target MLM weight should be none-zero!'

        self.init_weights()

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
        src_alignments,
        trg_alignments,
        include_clssep=True,
        return_dict=None,
        bert_base=None,
        src_mlm_input_ids=None,
        src_mlm_labels=None,
        src_mlm_special_word_masks=None,
        trg_mlm_input_ids=None,
        trg_mlm_labels=None,
        trg_mlm_special_word_masks=None,
    ):
        alignment_output = super().forward(
            src_input_ids,
            src_word_ids_lst,
            src_special_word_masks,
            trg_input_ids,
            trg_word_ids_lst,
            trg_special_word_masks,
            src_attention_masks,
            trg_attention_masks,
            src_alignments,
            trg_alignments,
            include_clssep,
            return_dict,
            bert_base,
        )
        alignment_mlm_loss = alignment_output.loss
        src_mlm_output = None
        trg_mlm_output = None

        if src_mlm_input_ids is not None and self.src_mlm_weight != 0.0:
            src_mlm_output = self._lm_forward(
                    src_mlm_input_ids,
                    src_attention_masks,
                    labels=src_mlm_labels,
                    return_dict=return_dict,
            )
            alignment_mlm_loss = alignment_mlm_loss \
                + src_mlm_output.loss * self.src_mlm_weight

        if trg_mlm_input_ids is not None and self.trg_mlm_weight != 0.0:
            trg_mlm_output = self._lm_forward(
                    trg_mlm_input_ids,
                    trg_attention_masks,
                    labels=trg_mlm_labels,
                    return_dict=return_dict,
            )
            alignment_mlm_loss = alignment_mlm_loss \
                + trg_mlm_output.loss * self.trg_mlm_weight

        output = CaoAlignmentMLMOutput(
            alignment_loss=alignment_output.alignment_loss,
            regularization_loss=alignment_output.regularization_loss,
            combined_alignment_loss=alignment_output.loss,
            src_hidden_states=alignment_output.src_hidden_states,
            trg_hidden_states=alignment_output.trg_hidden_states,
            src_mlm_output=src_mlm_output,
            trg_mlm_output=trg_mlm_output,
            loss=alignment_mlm_loss,
        )

        return output

    def _lm_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None \
                else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(
                    -1,
                    self.config.vocab_size
                ), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class CaoAlignmentOutput(ModelOutput):

    alignment_loss: Optional[torch.FloatTensor] = None
    regularization_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    src_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    trg_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CaoAlignmentMLMOutput(CaoAlignmentOutput):

    combined_alignment_loss: Optional[torch.FloatTensor] = None
    src_mlm_output: MaskedLMOutput = None
    trg_mlm_output: MaskedLMOutput = None


class SubwordToTokenStrategyBase():

    def process(self, features, word_ids_lsts, special_word_masks,
                include_clssep):
        filtered_word_ids_lsts = list()
        if include_clssep:
            filtered_word_ids_lsts = word_ids_lsts
        else:
            for lsts, masks in zip(word_ids_lsts, special_word_masks):
                pos = 0
                tmp_lst = list()
                for lst in lsts:
                    tmp_tmp_lst = list()
                    for idx, mask in zip(lst, masks[pos:pos+len(lst)]):
                        if mask.item() == 0:
                            tmp_tmp_lst.append(idx)

                    if len(tmp_tmp_lst) > 0:
                        tmp_lst.append(tmp_tmp_lst)
                    pos += len(lst)

                filtered_word_ids_lsts.append(tmp_lst)

        length = max([len(lst) for lst in filtered_word_ids_lsts])

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
            ids = list()
            pos = 0
            for id_lst in word_ids_lst:
                length = len(id_lst)
                masks = special_word_mask[pos:pos+length]
                if (include_clssep and masks[-1].item() in {0, 1}) \
                        or (not include_clssep and masks[-1].item() == 0):
                    ids.append(id_lst[-1])

                pos += length

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
        bert_base: PreTrainedModel = None,
        include_clssep: Optional[bool] = True,
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
        self.bert_base = bert_base
        for param in self.bert_base.parameters():
            assert not param.requires_grad
        self.include_clssep = include_clssep

    def get_train_dataloader(self):
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self._get_data_loader(
                self.train_dataset,
                self.args.train_batch_size,
                self.args.per_device_train_batch_size,
                'training',
                False,
        )

    def get_eval_dataloader(
            self,
            eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_datasets = eval_dataset if eval_dataset is not None else self.eval_dataset
        return self._get_data_loader(
                eval_datasets,
                self.args.eval_batch_size,
                self.args.per_device_eval_batch_size,
                'evaluation',
                True,
        )

    def get_test_dataloader(
            self,
            test_dataset: Dataset) -> DataLoader:
        return self._get_data_loader(
                test_dataset,
                self.args.eval_batch_size,
                self.args.per_device_eval_batch_size,
                'test',
                True,
        )

    def _get_data_loader(self, datasets, batch_size, per_device_batch_size,
                         description, eval=False):
        data_loaders = [
            self._get_single_dataloader(d,
                                        batch_size, per_device_batch_size,
                                        description, eval,
                                        )
            for d in datasets.datasets.values()
        ]

        if len(data_loaders) == 1:
            return data_loaders[0]

        return MultiDataLoader(
                datasets,
                data_loaders,
                batch_size,
        )

    def _get_single_dataloader(self, dataset, batch_size,
                               per_device_batch_size, description,
                               eval=False) -> DataLoader:

        if is_datasets_available() and isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)

        if isinstance(dataset, torch.utils.data.dataset.IterableDataset):
            if self.args.world_size > 1:
                dataset = IterableDatasetShard(
                    dataset,
                    batch_size=batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                dataset,
                batch_size=batch_size,
                #  collate_fn=self.data_collator,
                collate_fn=self.data_collator.get_eval() if eval else self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        #  sampler = self._get_single_sampler(
        #      dataset,
        #      batch_size,
        #      per_device_batch_size
        #  )
        #  return DataLoader(
        #      dataset,
        #      batch_size=batch_size,
        #      sampler=sampler,
        #      collate_fn=self.data_collator,
        #      drop_last=self.args.dataloader_drop_last,
        #      num_workers=self.args.dataloader_num_workers,
        #      pin_memory=self.args.dataloader_pin_memory,
        #  )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            #  collate_fn=self.data_collator,
            collate_fn=self.data_collator.get_eval() if eval else self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_single_sampler(self, dataset, batch_size, per_device_batch_size) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(dataset, Sized):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(dataset, Dataset):
                lengths = (
                    dataset[self.args.length_column_name]
                    if self.args.length_column_name in dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    dataset,
                    batch_size,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    dataset,
                    batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(dataset, generator=generator)
                return RandomSampler(dataset)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    dataset,
                    per_device_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.bert_base.device != model.device:
            self.bert_base = self.bert_base.to(model.device)

        outputs = model(bert_base=self.bert_base,
                        include_clssep=self.include_clssep, **inputs)

        if self.args.detailed_logging and self.state.global_step % self.args.logging_steps == 0:
            metrics = {
                'alignment_loss': outputs['alignment_loss'].item(),
                'regularization_loss': outputs['regularization_loss'].item(),
            }
            if type(model) == BertForCaoAlign:
                metrics['combined_alignment_loss'] = outputs['loss'].item()
            elif type(model) == BertForCaoAlignMLM:
                metrics['combined_alignment_loss'] = outputs['combined_alignment_loss'].item()
                metrics['src_mlm_loss'] = outputs['src_mlm_output']['loss'].item(
                ) if 'src_mlm_output' in outputs else 0.0
                metrics['trg_mlm_loss'] = outputs['trg_mlm_output']['loss'].item(
                ) if 'trg_mlm_output' in outputs else 0.0
                metrics['combined_alignment_mlm_loss'] = outputs['loss'].item()
            else:
                raise NotImplementedError(f'{type(model)} is not supported!')

            self.control = self.callback_handler.on_log(
                self.args,
                self.state,
                self.control,
                metrics
            )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        model_training = self.model.training
        self.model.eval()
        self.bert_base = self.bert_base.cpu()

        with torch.no_grad():
            if eval_dataset is not None:
                # mainly because of the non_context eval we need the full data in
                # one batch (based on the original implementation)
                eval_dataloader = self.get_eval_dataloader(eval_dataset)
            else:
                eval_dataloader = self.get_eval_dataloader()
                eval_dataset = self.eval_dataset

            res = dict()
            if type(eval_dataloader) == MultiDataLoader:
                for lang, dataset in tqdm(
                    zip(
                        eval_dataset.datasets.keys(),
                        eval_dataloader.data_loaders,
                    ),
                    desc='Eval languages',
                    total=len(eval_dataset.datasets)
                ):
                    for k, v in self._evaluate_single_dataset(dataset, lang).items():
                        res[f'{metric_key_prefix}_{lang}_{k}'] = v
            else:
                # this happens if only one dataset config (language) is given
                lang = list(eval_dataset.datasets.keys())[0]
                for k, v in self._evaluate_single_dataset(eval_dataloader, lang).items():
                    res[f'{metric_key_prefix}_{lang}_{k}'] = v

            if model_training:
                self.model.train()
            self.bert_base = self.bert_base.to(self.model.device)
            return res

    def _evaluate_single_dataset(self, dataloader, lang):
        res = dict()

        with tqdm(desc=lang, total=9) as pbar:
            ann_1 = None
            ann_2 = None
            src_alignments = list()
            trg_alignments = list()
            src_input_ids = None
            trg_input_ids = None
            sample_num = 0
            losses = dict()
            for input in dataloader:
                for a, b in zip(input['src_alignments'],
                                input['trg_alignments']):
                    src_alignments.append([a[0].item()+sample_num, a[1].item()])
                    trg_alignments.append([b[0].item()+sample_num, b[1].item()])
                if src_input_ids is None:
                    src_input_ids = input['src_input_ids']
                    trg_input_ids = input['trg_input_ids']
                else:
                    src_input_ids = cat_tensors_with_padding(src_input_ids, input['src_input_ids'])
                    trg_input_ids = cat_tensors_with_padding(trg_input_ids, input['trg_input_ids'])

                sample_num += input['src_input_ids'].shape[0]

                input = self._prepare_inputs(input)
                output = self.model(
                        bert_base=None,
                        return_dict=True,
                        **input
                )
                if ann_1 is None:
                    ann_1 = output['src_hidden_states']
                    ann_2 = output['trg_hidden_states']
                else:
                    ann_1 = cat_tensors_with_padding(ann_1, output['src_hidden_states'])
                    ann_2 = cat_tensors_with_padding(ann_2, output['trg_hidden_states'])

                for k, v in output.items():
                    if 'loss' in k:
                        losses.setdefault(k, list()).append(v.item())

            # FIXME for some reason output gets moved to the CPU
            ann_1 = ann_1.to(self.model.device)
            ann_2 = ann_2.to(self.model.device)
            src_alignments = torch.tensor(src_alignments, dtype=int).to(self.model.device)
            trg_alignments = torch.tensor(trg_alignments, dtype=int).to(self.model.device)
            del input
            del output
            torch.cuda.empty_cache()
            pbar.update(1)

            for k, v in self._evaluate_retrival_context(
                    src_alignments,
                    trg_alignments,
                    ann_1,
                    ann_2,
                    pbar).items():
                res[f'context_{k}'] = v

            for k, v in self._evaluate_retrival_noncontext(
                    src_input_ids,
                    trg_input_ids,
                    src_alignments,
                    trg_alignments,
                    ann_1,
                    ann_2,
                    pbar).items():
                res[f'non_context_{k}'] = v

            for k, v in losses.items():
                res[k] = np.mean(v)

        return res

    def _batch_eval_data(self, inputs, num_samples):
        bs = self.args.per_device_eval_batch_size
        for i in range(0, num_samples, bs):
            yield {
                k: v[i:i+bs]
                for k, v in inputs.items()
            }

    def _evaluate_retrival(self, ann_1, ann_2, pbar):
        hub_1, hub_2 = hubness_CSLS(ann_1, ann_2)
        pbar.update(1)

        matches_1 = [
            bestk_idx_CSLS(ann, ann_2, hub_2)[0].detach().cpu().numpy()
            for ann in ann_1
        ]
        pbar.update(1)

        matches_2 = [
            bestk_idx_CSLS(ann, ann_1, hub_1)[0].detach().cpu().numpy()
            for ann in ann_2
        ]
        pbar.update(1)

        acc_1 = np.sum(
            np.array(matches_1) == np.arange(len(matches_1))
        ) / len(matches_1)
        acc_2 = np.sum(
            np.array(matches_2) == np.arange(len(matches_2))
        ) / len(matches_2)
        pbar.update(1)

        return {
            'src2trg': acc_1,
            'trg2src': acc_2,
        }

    def _evaluate_retrival_context(self, src_alignments, trg_alignments, ann_1,
                                   ann_2, pbar):
        # gives exactly the same results for some language pairs but there are
        # some slight differences for others. Good enough though
        ann_1 = ann_1[src_alignments[:, 0], src_alignments[:, 1]]
        ann_2 = ann_2[trg_alignments[:, 0], trg_alignments[:, 1]]

        return self._evaluate_retrival(
            ann_1,
            ann_2,
            pbar,
        )

    def _evaluate_retrival_noncontext(self, src_input_ids, trg_input_ids,
                                      src_alignments, trg_alignments, ann_1,
                                      ann_2, pbar):
        # XXX does not give the exact results as the original implementation
        # that should be used instead but this is also good to do quick checks
        src_sentences = detokenize(src_input_ids, self.tokenizer)
        trg_sentences = detokenize(trg_input_ids, self.tokenizer)

        final_1 = list()
        final_2 = list()
        found_1 = list()
        found_2 = list()
        final_idx_1 = list()
        final_idx_2 = list()
        for src, trg in zip(src_alignments, trg_alignments):
            final_1.append(src_sentences[src[0]][src[1]])
            final_2.append(trg_sentences[trg[0]][trg[1]])

        for i in range(len(final_1)):
            if final_1[i] not in found_1:
                found_1.append(final_1[i])
                found_2.append(final_2[i])
                final_idx_1.append(src_alignments[i].cpu().numpy())
                final_idx_2.append(trg_alignments[i].cpu().numpy())

        final_idx_1 = torch.tensor(
            final_idx_1, dtype=src_alignments.dtype).to(src_alignments.device)
        final_idx_2 = torch.tensor(
            final_idx_2, dtype=trg_alignments.dtype).to(trg_alignments.device)

        ann_1 = ann_1[final_idx_1[:, 0], final_idx_1[:, 1]]
        ann_2 = ann_2[final_idx_2[:, 0], final_idx_2[:, 1]]

        return self._evaluate_retrival(
            ann_1,
            ann_2,
            pbar,
        )

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self.log(metrics)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
