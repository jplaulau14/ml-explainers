from functools import partial
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lora_explainer.config import RunConfig

PROMPT_HEAD = "Review:"
PROMPT_TAIL = "\nSentiment:"
TARGETS = (" negative", " positive")
IGNORE = -100

Item = dict[str, Any]


def load_tokenizer(model_id: str, revision: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def target_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    encoded = [tokenizer.encode(target) for target in TARGETS]
    assert all(len(ids) == 1 for ids in encoded)
    return [ids[0] for ids in encoded]


def prompt_ids(tokenizer: PreTrainedTokenizerBase, sentence: str, budget: int) -> list[int]:
    head = tokenizer.encode(PROMPT_HEAD)
    tail = tokenizer.encode(PROMPT_TAIL)
    body = tokenizer.encode(" " + sentence.strip())
    return head + body[: budget - len(head) - len(tail)] + tail


def make_item(
    tokenizer: PreTrainedTokenizerBase, row: Item, max_length: int, answers: list[int] | None
) -> Item:
    prompt = prompt_ids(tokenizer, row["sentence"], max_length - 1)
    answer = [answers[row["label"]]] if answers else []
    labels = [IGNORE] * len(prompt) + answer
    return {"input_ids": prompt + answer, "labels": labels, "label": row["label"]}


def pad(rows: list[list[int]], value: int) -> torch.Tensor:
    width = max(len(row) for row in rows)
    return torch.tensor([row + [value] * (width - len(row)) for row in rows])


def collate(items: list[Item], pad_id: int) -> dict[str, torch.Tensor]:
    ids = [item["input_ids"] for item in items]
    return {
        "input_ids": pad(ids, pad_id),
        "attention_mask": pad([[1] * len(row) for row in ids], 0),
        "labels": pad([item["labels"] for item in items], IGNORE),
        "label": torch.tensor([item["label"] for item in items]),
    }


def load_split(config: RunConfig, revision: str, split: str) -> Dataset:
    return load_dataset(config.dataset_id, split=split, revision=revision)


def make_loader(
    tokenizer: PreTrainedTokenizerBase, rows: Dataset, config: RunConfig, with_target: bool
) -> DataLoader:
    answers = target_ids(tokenizer) if with_target else None
    items = [make_item(tokenizer, row, config.max_length, answers) for row in rows]
    pad_with = partial(collate, pad_id=tokenizer.pad_token_id)
    return DataLoader(items, batch_size=config.batch_size, shuffle=False, collate_fn=pad_with)


def train_loader(
    tokenizer: PreTrainedTokenizerBase, config: RunConfig, revision: str
) -> DataLoader:
    rows = load_split(config, revision, "train").shuffle(seed=config.seed)
    return make_loader(tokenizer, rows.select(range(config.train_examples)), config, True)


def validation_loader(
    tokenizer: PreTrainedTokenizerBase, config: RunConfig, revision: str
) -> DataLoader:
    return make_loader(tokenizer, load_split(config, revision, "validation"), config, False)
