import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .modules import *

from typing import Optional


class BertForMultiLabelCls(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        assert (
            args.cls_head != "attention"
        ), f"BertForMultiLabelCls does not implement AttentionClsHead"

        self.bert = AutoModel.from_pretrained(
            args.model_name_or_path, local_files_only=args.use_local
        )

        cls_head_class = get_cls_head(args.cls_head)
        self.cls_head = cls_head_class(**vars(args))

    def forward(self, **batch):
        bert_output = self.bert(**batch).last_hidden_state
        bert_output = bert_output[:, 0, :]  # Take output of CLS token
        logits = self.cls_head(bert_output)
        return logits, _


class BertAttentionForMultiLabelCls(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        assert (
            args.cls_head == "attention"
        ), f"BertAttentionForMultiLabelCls only implement AttentionClsHead"

        self.bert = AutoModel.from_pretrained(
            args.model_name_or_path, local_files_only=args.use_local
        )

        cls_head_class = get_cls_head(args.cls_head)
        self.cls_head = cls_head_class(**vars(args))

    def forward(self, **batch):
        bert_output = self.bert(**batch).last_hidden_state
        logits = self.cls_head(bert_output)
        return logits, _


class BertSequenceForMultiLabelCls(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        assert (
            args.cls_head != "attention"
        ), f"BertSequenceForMultiLabelCls does not implement AttentionClsHead"

        self.bert = AutoModel.from_pretrained(
            args.model_name_or_path, local_files_only=args.use_local
        )

        sequence_aggregator_class = get_sequence_aggregator(args.sequence_strategy)
        self.aggregator = sequence_aggregator_class(**vars(args))

        cls_head_class = get_cls_head(args.cls_head)
        self.cls_head = cls_head_class(**vars(args))

    def forward(self, **batch):
        # Input shape: batch_size x num_chunk x max_seq_len
        # Feed each chunk into bert and stack to obtain sequence

        bs, num_chunk, max_seq_len = batch["input_ids"].shape

        all_bert_outputs = []
        for i in range(num_chunk):
            batch_chunk = {k: v[:, i, :] for k, v in batch.items()}
            bert_output = self.bert(**batch_chunk).last_hidden_state
            bert_output = bert_output[:, 0, :]  # Take output of CLS token
            all_bert_outputs.append(bert_output)

        bert_sequence_output = torch.stack(all_bert_outputs, dim=1)
        hiddens = self.aggregator(bert_sequence_output)  # B x H
        logits = self.cls_head(hiddens)
        return logits, _


class BertSequenceAttentionForMultiLabelCls(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        assert args.sequence_strategy not in [
            "average",
            "sum",
        ], f'{args.sequence_strategy} must be either "cnn" or "rnn"'

        self.bert = AutoModel.from_pretrained(
            args.model_name_or_path, local_files_only=args.use_local
        )

        sequence_aggregator_class = get_sequence_aggregator(args.sequence_strategy)
        self.aggregator = sequence_aggregator_class(**vars(args))

        cls_head = "attention"
        cls_head_class = get_cls_head(cls_head)
        self.cls_head = cls_head_class(**vars(args))

    def forward(self, **batch):
        # Input shape: batch_size x num_chunk x max_seq_len
        # Feed each chunk into bert and stack to obtain sequence

        bs, num_chunk, max_seq_len = batch["input_ids"].shape

        all_bert_outputs = []
        for i in range(num_chunk):
            batch_chunk = {k: v[:, i, :] for k, v in batch.items()}
            bert_output = self.bert(**batch_chunk).last_hidden_state
            bert_output = bert_output[:, 0, :]  # Take output of CLS token
            all_bert_outputs.append(bert_output)

        bert_sequence_output = torch.stack(all_bert_outputs, dim=1)
        hiddens = self.aggregator(
            bert_sequence_output, return_sequence=True
        )  # B x S x H
        logits = self.cls_head(hiddens)
        return logits, _
