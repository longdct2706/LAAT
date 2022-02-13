from .truncate_processor import (
    TRUNCATE_PROCESSOR_TYPES,
    HeadTruncateProcessor,
    TailTruncateProcessor,
    HeadTailTruncateProcessor,
)
from .sequence_processor import (
    CHUNK_PROCESSOR_TYPES,
    ModelChunkProcessor,
    SentenceChunkProcessor,
)

DATASET_PROCESSOR_TYPES = TRUNCATE_PROCESSOR_TYPES + CHUNK_PROCESSOR_TYPES


def get_dataset_processor(dataset_processor: str):
    assert (
        dataset_processor in DATASET_PROCESSOR_TYPES
    ), f"{dataset_processor} is not implemented"
    if dataset_processor == "head":
        return HeadTruncateProcessor
    elif dataset_processor == "tail":
        return TailTruncateProcessor
    elif dataset_processor == "head_tail":
        return HeadTailTruncateProcessor
    elif dataset_processor == "model_chunk":
        return ModelChunkProcessor
    elif dataset_processor == "sentence_chunk":
        return SentenceChunkProcessor
    else:
        raise NotImplementedError
