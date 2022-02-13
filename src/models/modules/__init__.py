from .classification_head import get_cls_head, SimpleClsHead, ClsHead, AttentionClsHead
from .sequence_aggregator import (
    get_sequence_aggregator,
    AverageAggregator,
    SumAggregator,
    CNNAggregator,
    RNNAggregator,
)
