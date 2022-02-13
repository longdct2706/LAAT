import math
from pathlib import Path
import re

from transformers import AutoTokenizer
from src.data_helpers.dataset_processor.utils import (
    remove_cls_sep_tokens,
    truncate_list,
    pad_list,
)

TRUNCATE_PROCESSOR_TYPES = ["head", "tail", "head_tail"]
CHUNK_PROCESSOR_TYPES = ["model_chunk", "sentence_chunk"]
DATASET_PROCESSOR_TYPES = TRUNCATE_PROCESSOR_TYPES + CHUNK_PROCESSOR_TYPES


def path_to_str(path: Path) -> str:
    return str(path.resolve())


def replace_multiple_newline(text: str) -> str:
    return re.sub("\n+", "\n", text)


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


class DocumentProcessor:
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_len: int = 512,
        use_local: bool = False,
        **kwargs,
    ):
        super().__init__()

        # self.dims = (1, 28, 28)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, local_files_only=use_local
        )
        self.max_seq_len = max_seq_len

    def process_document(self, document: str) -> dict:
        model_input = self.tokenizer(
            document,
            return_tensors="np",
        )
        model_input = remove_cls_sep_tokens(
            model_input, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        )

        input_ids = model_input["input_ids"]
        token_type_ids = model_input["token_type_ids"]
        attention_mask = model_input["attention_mask"]

        input_ids = self._process_input_ids(input_ids)
        token_type_ids = self._process_token_type_ids(token_type_ids)
        attention_mask = self._process_attention_mask(attention_mask)

        model_input = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

        return model_input

    def _process_input_ids(self, input_ids: list) -> list:
        pass

    def _process_token_type_ids(self, token_type_ids: list) -> list:
        pass

    def _process_attention_mask(self, attention_mask: list) -> list:
        pass


class HeadTruncateProcessor(DocumentProcessor):
    def _process_input_ids(self, input_ids: list) -> list:
        input_ids = truncate_list(input_ids, self.max_seq_len - 2, mode="head")
        input_ids = (
            [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        )
        input_ids = pad_list(
            input_ids, self.max_seq_len, value=self.tokenizer.pad_token_id
        )
        return input_ids

    def _process_token_type_ids(self, token_type_ids: list) -> list:
        token_type_ids = truncate_list(
            token_type_ids, self.max_seq_len - 2, mode="head"
        )
        token_type_ids = [0] + token_type_ids + [0]
        token_type_ids = pad_list(token_type_ids, self.max_seq_len, value=0)
        return token_type_ids

    def _process_attention_mask(self, attention_mask: list) -> list:
        attention_mask = truncate_list(
            attention_mask, self.max_seq_len - 2, mode="head"
        )
        attention_mask = [1] + attention_mask + [1]
        attention_mask = pad_list(attention_mask, self.max_seq_len, value=0)
        return attention_mask


class TailTruncateProcessor(DocumentProcessor):
    def _process_input_ids(self, input_ids: list) -> list:
        input_ids = truncate_list(input_ids, self.max_seq_len - 2, mode="tail")
        input_ids = (
            [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        )
        input_ids = pad_list(
            input_ids, self.max_seq_len, value=self.tokenizer.pad_token_id
        )
        return input_ids

    def _process_token_type_ids(self, token_type_ids: list) -> list:
        token_type_ids = truncate_list(
            token_type_ids, self.max_seq_len - 2, mode="tail"
        )
        token_type_ids = [0] + token_type_ids + [0]
        token_type_ids = pad_list(token_type_ids, self.max_seq_len, value=0)
        return token_type_ids

    def _process_attention_mask(self, attention_mask: list) -> list:
        attention_mask = truncate_list(
            attention_mask, self.max_seq_len - 2, mode="tail"
        )
        attention_mask = [1] + attention_mask + [1]
        attention_mask = pad_list(attention_mask, self.max_seq_len, value=0)
        return attention_mask


class HeadTailTruncateProcessor(HeadTruncateProcessor):
    def __init__(
        self,
        tokenizer,
        file_paths,
        max_seq_len: int = 512,
        label_col: str = "Full_Labels",
        head_ratio: float = 0.25,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            file_paths=file_paths,
            max_seq_len=max_seq_len,
            label_col=label_col,
            **kwargs,
        )
        self.head_ratio = head_ratio
        self.head_length = math.floor((self.max_seq_len - 3) * self.head_ratio)
        self.tail_length = self.max_seq_len - 3 - self.head_length

    def _process_input_ids(self, input_ids: list) -> list:
        if len(input_ids) < self.max_seq_len - 3:
            return super()._process_input_ids(input_ids)

        head_input_ids = truncate_list(input_ids, self.head_length, mode="head")
        tail_input_ids = truncate_list(input_ids, self.tail_length, mode="tail")

        input_ids = (
            [self.tokenizer.cls_token_id]
            + head_input_ids
            + [self.tokenizer.sep_token_id]
            + tail_input_ids
            + [self.tokenizer.sep_token_id]
        )
        return input_ids

    def _process_token_type_ids(self, token_type_ids: list) -> list:
        if len(token_type_ids) < self.max_seq_len - 3:
            return super()._process_token_type_ids(token_type_ids)

        token_type_ids = (
            [0]  # CLS token
            + [0] * self.head_length  # First sentence
            + [1]  # SEP token
            + [1] * self.tail_length  # Second sentence
            + [1]  # SEP token
        )
        return token_type_ids

    def _process_attention_mask(self, attention_mask: list) -> list:
        if len(attention_mask) < self.max_seq_len - 3:
            return super()._process_attention_mask(attention_mask)

        # Since head + tail will span to maximum max_seq_len -> no need to pad
        attention_mask = [1] * self.max_seq_len
        return attention_mask


class ModelChunkProcessor(DocumentProcessor):
    def _process_input_ids(self, input_ids: list) -> list:
        input_ids_chunks = []

        seq_len = self.max_seq_len - 2
        num_chunks = math.ceil(len(input_ids) / seq_len)
        for i in range(num_chunks):
            sub_input_ids = input_ids[i * seq_len : (i + 1) * seq_len]
            sub_input_ids = (
                [self.tokenizer.cls_token_id]
                + sub_input_ids
                + [self.tokenizer.sep_token_id]
            )
            input_ids_chunks.append(sub_input_ids)

        input_ids_chunks[-1] = pad_list(
            input_ids_chunks[-1], self.max_seq_len, value=self.tokenizer.pad_token_id
        )
        return input_ids_chunks

    def _process_token_type_ids(self, token_type_ids: list) -> list:
        token_type_ids_chunk = []
        seq_len = self.max_seq_len - 2
        num_chunks = math.ceil(len(token_type_ids) / seq_len)

        for _ in range(num_chunks):
            token_type_ids_chunk.append([0] * self.max_seq_len)

        return token_type_ids_chunk

    def _process_attention_mask(self, attention_mask: list) -> list:
        attention_mask_chunk = []
        seq_len = self.max_seq_len - 2
        num_chunks = math.ceil(len(attention_mask) / seq_len)
        num_tokens_last_chunk = len(attention_mask) % seq_len
        num_tokens_need_pad = seq_len - num_tokens_last_chunk

        for _ in range(num_chunks - 1):
            attention_mask_chunk.append([1] * self.max_seq_len)
        attention_mask_chunk.append(
            [1]  # CLS token
            + [1] * num_tokens_last_chunk
            + [1]  # SEP token
            + [0] * num_tokens_need_pad
        )
        return attention_mask_chunk


class SentenceChunkProcessor(DocumentProcessor):
    def __init__(
        self,
        tokenizer,
        file_paths,
        max_seq_len: int = 512,
        label_col: str = "Full_Labels",
        num_sents_overlap: int = 0,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            file_paths=file_paths,
            max_seq_len=max_seq_len,
            label_col=label_col,
            **kwargs,
        )
        self.num_sents_overlap = num_sents_overlap

    def process_document(self, document: str) -> dict:
        document = replace_multiple_newline(document)
        sentences = document.split("\n")
        all_input_ids = []

        for sent in sentences:
            model_input = self.tokenizer(
                sent,
                return_tensors="np",
            )
            input_ids = remove_cls_sep_tokens(
                model_input,
                cls_id=self.tokenizer.cls_token_id,
                sep_id=self.tokenizer.sep_token_id,
            )["input_ids"]
            if len(input_ids) >= self.max_seq_len - 2:
                input_ids = input_ids[: self.max_seq_len - 2]
                input_ids = input_ids + [self.tokenizer.sep_token_id]
            all_input_ids.append(input_ids)

        chunk_input_ids = []
        current_chunk = []
        for i, input_ids in enumerate(all_input_ids):
            if len(current_chunk) + len(input_ids) >= self.max_seq_len:
                chunk_input_ids.append(current_chunk)
                current_chunk = []
                if self.num_sents_overlap > 0:
                    current_chunk = all_input_ids[i - self.num_sents_overlap : i]
            current_chunk += input_ids
        chunk_input_ids.append(current_chunk)

        output_input_ids = []
        output_attention_mask = []
        output_token_type_ids = []
        for chunk in chunk_input_ids:
            chunk_with_cls = [self.tokenizer.cls_token_id] + chunk
            pad_len = self.max_seq_len - len(chunk_with_cls)

            padding = [self.tokenizer.pad_token_id] * pad_len
            chunk = chunk_with_cls + padding
            output_input_ids.append(chunk)

            attention_mask = [1] * len(chunk_with_cls) + [0] * pad_len
            output_attention_mask.append(attention_mask)

            token_type_ids = [0] * self.max_seq_len
            output_token_type_ids.append(token_type_ids)

        model_input = {
            "input_ids": output_input_ids,
            "token_type_ids": output_token_type_ids,
            "attention_mask": output_attention_mask,
        }

        return model_input


class LabelProcessor(object):
    def __init__(
        self,
        training_labels: list,
    ):
        self.label2index = []
        self.index2label = []

        self.all_labels = []

        self.logger = None
        self.update_labels(training_labels)

    def index_of_label(self, label: str, level: int) -> int:
        try:
            return self.label2index[level][label]
        except:
            return 0

    def update_labels(self, labels):
        self.all_labels = []
        self.index2label = []
        self.label2index = []
        for level_labels in labels:
            all_labels = list(sorted(level_labels))
            self.label2index.append(
                {label: idx for idx, label in enumerate(all_labels)}
            )
            self.index2label.append(
                {idx: label for idx, label in enumerate(all_labels)}
            )
            self.all_labels.append(all_labels)

    def n_labels(self, level):
        return len(self.all_labels[level])

    def n_level(self):
        return len(self.all_labels)

    def all_n_labels(self):
        output = []
        for level in range(len(self.all_labels)):
            output.append(len(self.all_labels[level]))
        return output
