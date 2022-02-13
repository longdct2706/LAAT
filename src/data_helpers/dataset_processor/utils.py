from itertools import chain
import numpy as np


def remove_cls_sep_tokens(model_input: dict, cls_id: int, sep_id: int) -> dict:
    input_ids = model_input["input_ids"]
    indices_to_remove = np.zeros(len(input_ids))
    indices_to_remove = np.logical_or(indices_to_remove, input_ids == cls_id)
    indices_to_remove = np.logical_or(indices_to_remove, input_ids == sep_id)
    return {k: v[~indices_to_remove].tolist() for k, v in model_input.items()}


def truncate_list(input_list: list, max_seq_len: int, mode: str = "head") -> list:
    if len(input_list) <= max_seq_len:
        return input_list
    if mode == "head":
        return input_list[:max_seq_len]
    elif mode == "tail":
        return input_list[-max_seq_len:]
    else:
        raise NotImplementedError(f"{mode} truncation is not implemented")


def pad_list(input_list: list, max_seq_len: int, value: int) -> list:
    if len(input_list) >= max_seq_len:
        return input_list
    return input_list + [value] * (max_seq_len - len(input_list))


def flatten_list(input_list: list) -> list:
    return list(chain.from_iterable(input_list))
