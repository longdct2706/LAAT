import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQUENCE_AGGREGATOR_TYPES = [
    "average",
    "sum",
    "cnn",
    "rnn",
    "sequence",
]


def get_sequence_aggregator(sequence_strategy: str):
    assert (
        sequence_strategy in SEQUENCE_AGGREGATOR_TYPES
    ), f"{sequence_strategy} is not implemented"
    if sequence_strategy == "average":
        return AverageAggregator
    elif sequence_strategy == "sum":
        return SumAggregator
    elif sequence_strategy == "cnn":
        return CNNAggregator
    elif sequence_strategy == "rnn":
        return RNNAggregator
    elif sequence_strategy == "sequence":
        return SequenceAggregator


class AverageAggregator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, sequence, return_sequence: bool = False, **kwargs):
        # Sequence shape: B x S x H
        out = torch.mean(sequence, dim=1)
        if return_sequence:
            out = out.unsqueeze(dim=1)
        return out


class SumAggregator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, sequence, return_sequence: bool = False, **kwargs):
        # Sequence shape: B x S x H
        out = torch.sum(sequence, dim=1)
        if return_sequence:
            out = out.unsqueeze(dim=1)
        return out


class SequenceAggregator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, sequence, return_sequence: bool = False, **kwargs):
        # Sequence shape: B x S x H
        if return_sequence:
            return self.identity(sequence)
        else:
            return self.identity(sequence[:, 0, :])


class CNNAggregator(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int = 3, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=int(math.floor(kernel_size / 2)),
        )

    def forward(self, sequence, return_sequence: bool = False, **kwargs):
        # Sequence shape: B x S x H
        sequence = sequence.transpose(1, 2)  # B x S x H -> B x H x S
        feature_map = self.conv(sequence)
        feature_map = F.relu(feature_map)
        if return_sequence:
            return feature_map.transpose(1, 2)  # B x H x S -> B x S x H
        else:
            return F.max_pool1d(feature_map, feature_map.size(-1)).squeeze(-1)


class RNNAggregator(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        **kwargs,
    ):
        super().__init__()

        if bidirectional:
            output_size = hidden_size // 2
        else:
            output_size = hidden_size

        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=output_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, sequence, return_sequence: bool = False, **kwargs):
        out, _ = self.rnn(sequence)
        if return_sequence:
            return out
        else:
            return out[:, -1, :]  # Last element
