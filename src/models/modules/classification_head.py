import torch
import torch.nn as nn

from typing import Optional

CLS_HEAD_TYPES = ["simple", "complex", "attention"]


def get_cls_head(cls_head: str):
    assert cls_head in CLS_HEAD_TYPES, f"{cls_head} is not implemented"
    if cls_head == "simple":
        return SimpleClsHead
    elif cls_head == "complex":
        return ClsHead
    elif cls_head == "attention":
        return AttentionClsHead
    else:
        return None


class SimpleClsHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self, hidden_size: int, num_labels: int, dropout: float = 0.0, **kawrgs
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_labels, bias=True)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ClsHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self, hidden_size: int, num_labels: int, dropout: float = 0.0, **kawrgs
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AttentionClsHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        attention_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if attention_size is None:
            attention_size = hidden_size
        self.num_labels = num_labels

        self.fc1 = nn.Linear(hidden_size, attention_size, bias=False)
        self.fc2 = nn.Linear(attention_size, num_labels, bias=False)
        self.fc_output = nn.Linear(hidden_size, num_labels, bias=True)
        self._init_weights(mean=0.0, std=0.03)

    def _init_weights(self, mean=0.0, std=0.03) -> None:
        """
        Initialise the weights
        :param mean:
        :param std:
        :return: None
        """
        torch.nn.init.normal(self.fc1.weight, mean, std)
        if self.fc1.bias is not None:
            self.fc1.bias.data.fill_(0)
        torch.nn.init.normal(self.fc2.weight, mean, std)
        if self.fc2.bias is not None:
            self.fc2.bias.data.fill_(0)
        torch.nn.init.normal(self.fc_output.weight, mean, std)
        if self.fc_output.bias is not None:
            self.fc_output.bias.data.fill_(0)

    def forward(self, x, **kwargs):
        # Input shape: B x S x H
        projection = torch.tanh(self.fc1(x))  # B x S x A
        attention_weight = torch.softmax(self.fc2(projection), dim=1)  # B x S x L
        attention_weight = attention_weight.transpose(1, 2)  # B x L x S
        weighted_output = attention_weight @ x  # B x L x H
        logits = (
            self.fc_output.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.fc_output.bias)
        )
        return logits
