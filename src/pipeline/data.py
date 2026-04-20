from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Record:
    sequence: str
    label: int


class SequenceDataset(Dataset):
    def __init__(self, rows: List[Record]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Record:
        return self.rows[idx]


def read_split_rows(path: Path) -> Dict[str, List[Record]]:
    splits: Dict[str, List[Record]] = {"train": [], "val": [], "test": []}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split not in splits:
                raise ValueError(f"Unexpected split '{split}'. Expected one of: {list(splits)}")
            splits[split].append(Record(sequence=row["sequence"], label=int(row["label"])))
    return splits


def validate_splits(splits: Dict[str, List[Record]], source: Path) -> None:
    for name in ("train", "val", "test"):
        if len(splits[name]) == 0:
            raise ValueError(f"Split '{name}' is empty in {source}")


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch: List[Record]) -> Dict[str, torch.Tensor]:
        sequences = [x.sequence for x in batch]
        labels = torch.tensor([x.label for x in batch], dtype=torch.long)
        tok = tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok["labels"] = labels
        return tok

    return collate


def build_dataloaders(
    splits: Dict[str, List[Record]],
    tokenizer,
    batch_size: int,
    max_length: int,
) -> Dict[str, DataLoader]:
    collate_fn = make_collate_fn(tokenizer=tokenizer, max_length=max_length)
    return {
        "train": DataLoader(
            SequenceDataset(splits["train"]),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            SequenceDataset(splits["val"]),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            SequenceDataset(splits["test"]),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),
    }

