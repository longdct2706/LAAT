# -*- coding: utf-8 -*-
"""
    This provides the functions to load the data for training and testing the model (e.g., batch)
    Author: Thanh Vu <thanh.vu@csiro.au>
    Date created: 01/03/2019
    Date last modified: 19/08/2020
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm
import numpy as np

from src.util.preprocessing import SENTENCE_SEPARATOR, RECORD_SEPARATOR
from src.data_helpers.bert_processor import DocumentProcessor, LabelProcessor


class BertTextDataset(Dataset):
    def __init__(
        self,
        text_data: list,
        text_processor: DocumentProcessor,
        label_processor: LabelProcessor,
        max_seq_length: int = 512,
        multilabel: bool = False,
        **kwargs
    ):
        super().__init__()
        self.text_processor = text_processor
        self.label_processor = label_processor
        self.multilabel = multilabel
        self.max_seq_length = max_seq_length
        self.indexed_data = []
        self.n_instances = len(text_data)
        self.n_total_tokens = 0

        n_label_level = len(text_data[0][1])
        self.label_count = [dict() for _ in range(n_label_level)]
        self.labels = [set() for _ in range(n_label_level)]
        for text, labels, _id in tqdm(
            text_data, unit="samples", desc="Processing data"
        ):
            label_list = [[] for _ in range(n_label_level)]
            if type(labels) == list:
                for label_lvl in range(len(labels)):
                    for label in labels[label_lvl]:
                        if label in self.label_processor.label2index[label_lvl]:

                            label = self.label_processor.index_of_label(
                                label, label_lvl
                            )
                            if label not in self.label_count[label_lvl]:
                                self.label_count[label_lvl][label] = 1
                            else:
                                self.label_count[label_lvl][label] += 1
                            self.labels[label_lvl].add(label)
                            label_list[label_lvl].append(label)
                        else:
                            continue

            if len(label_list) == 0:
                continue

            is_skipped = False
            for level_label in label_list:
                if len(level_label) == 0:
                    is_skipped = True
                    break
            if is_skipped:
                continue

            model_inputs = self.text_processor.process_document(text)
            model_inputs["labels"] = label_list
            model_inputs["id"] = _id

            # after processing all records
            self.indexed_data.append(model_inputs)

        self.labels = sorted(list(self.labels))
        self.size = len(self.indexed_data)
        self._reprocess_labels()

    def shuffle_data(self):
        random.shuffle(self.indexed_data)

    def _reprocess_labels(self):
        for example in self.indexed_data:
            label_list = example["labels"]
            if not self.multilabel:
                label_out = [None for _ in range(len(label_list))]
                for idx in range(len(label_list)):
                    label_out[idx] = label_list[idx][0]
                example["labels"] = label_out
            else:
                all_one_hot_label_list = []
                for label_lvl in range(len(label_list)):
                    one_hot_label_list = [0] * self.label_processor.n_labels(label_lvl)
                    for label in label_list[label_lvl]:
                        one_hot_label_list[label] = 1
                    all_one_hot_label_list.append(
                        np.asarray(one_hot_label_list).astype(np.int32)
                    )
                example["labels"] = all_one_hot_label_list

    def __getitem__(self, index):
        return self.indexed_data[index]

    def __len__(self):
        return len(self.indexed_data)


class BertTextDataLoader(DataLoader):
    def __init__(self, pad_token_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.pad_token_id = pad_token_id

    def _collate_fn(self, batch):
        is_sequence = isinstance(batch[0]["input_ids"][0], list)

        if is_sequence:
            return self._collate_fn_sequence(batch)
        else:
            return self._collate_fn_single(batch)

    def _collate_fn_sequence(self, batch):
        def pad_list_of_list(list_of_list, target_size_outer_list, value):
            current_size_outer_list = len(list_of_list)
            size_inner_list = len(list_of_list[0])
            size_to_pad = target_size_outer_list - current_size_outer_list
            pad_list = [[value] * size_inner_list]
            pad_list = pad_list * size_to_pad
            return list_of_list + pad_list

        def pad_and_batch(_batch, value):
            longest_sequence = max([len(example) for example in _batch])
            _batch = [
                pad_list_of_list(example, longest_sequence, value) for example in _batch
            ]
            return torch.LongTensor(_batch)

        # List of dict -> Dict of list
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        input_ids_batch = pad_and_batch(batch["input_ids"], self.pad_token_id)
        attention_mask_batch = pad_and_batch(batch["attention_mask"], 0)
        token_type_ids_batch = pad_and_batch(batch["token_type_ids"], 0)
        labels_batch = self._process_label_batch(batch["labels"])
        id_batch = torch.LongTensor(batch["id"])

        batch = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "token_type_ids": token_type_ids_batch,
            "labels": labels_batch,
            "id": id_batch,
        }

        return batch

    def _collate_fn_single(self, batch):
        # List of dict -> Dict of list
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        input_ids_batch = torch.LongTensor(batch["input_ids"])
        attention_mask_batch = torch.LongTensor(batch["attention_mask"])
        token_type_ids_batch = torch.LongTensor(batch["token_type_ids"])
        labels_batch = self._process_label_batch(batch["labels"])
        id_batch = torch.LongTensor(batch["id"])

        batch = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "token_type_ids": token_type_ids_batch,
            "labels": labels_batch,
            "id": id_batch,
        }

        return batch

    def _process_label_batch(self, batch):
        multilabel = isinstance(batch[0], np.ndarray)
        label_batch = np.stack(batch, axis=0)
        norm_label_batch = []
        if not multilabel:
            for label_lvl in range(label_batch.shape[1]):
                norm_label_batch.append(
                    torch.LongTensor(label_batch[:, label_lvl].tolist())
                )
        else:
            for label_lvl in range(label_batch.shape[1]):
                norm_label_batch.append(
                    torch.FloatTensor(label_batch[:, label_lvl].tolist())
                )
        return norm_label_batch
