#!/usr/bin/env python3
"""
Export splice donor windows from the revised Nucleotide Transformer benchmark
to the pipeline CSV format: sequence, label, split.

Source: https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised

The hub combines all tasks; donor rows use task id ``splice_sites_donors`` (plural).
HF splits are train / test with chromosome-held-out test; this script keeps the
official test split and draws a stratified validation subset from train so
``read_split_rows`` gets non-empty train/val/test buckets.

Requires: pip install datasets
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

TASK_ID = "splice_sites_donors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/splice_donors_nt_revised.csv"),
        help="Output CSV path (parent dirs are created).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the train→val carve.")
    p.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction of official *train* rows assigned to val (stratified by label).",
    )
    p.add_argument(
        "--train_max",
        type=int,
        default=None,
        help="Cap train rows after split (shuffled subsample). Use for fast smoke tests.",
    )
    p.add_argument(
        "--val_max",
        type=int,
        default=None,
        help="Cap val rows (shuffled subsample).",
    )
    p.add_argument(
        "--test_max",
        type=int,
        default=None,
        help="Cap test rows (shuffled subsample). Official test is still chrom-held-out before capping.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Short preset: --train_max 512 --val_max 128 --test_max 128 and default out "
        "data/processed/splice_donors_smoke.csv (overrides --out unless you set --out).",
    )
    return p.parse_args()


def _stratified_train_val_indices(labels: Sequence[int], val_frac: float, rng: random.Random) -> Tuple[List[int], List[int]]:
    by_class: Dict[int, List[int]] = {0: [], 1: []}
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for y in (0, 1):
        idxs = by_class[y]
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _cap_indices(idxs: List[int], max_n: int | None, rng: random.Random) -> List[int]:
    """Shuffle subsample index list for smaller / faster runs."""
    if max_n is None or len(idxs) <= max_n:
        return idxs
    pool = list(idxs)
    rng.shuffle(pool)
    return pool[:max_n]


def main() -> int:
    args = parse_args()
    if args.smoke:
        if args.train_max is None:
            args.train_max = 512
        if args.val_max is None:
            args.val_max = 128
        if args.test_max is None:
            args.test_max = 128
        if str(args.out) == "data/processed/splice_donors_nt_revised.csv":
            args.out = Path("data/processed/splice_donors_smoke.csv")
    rng = random.Random(args.seed)

    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Missing dependency: pip install datasets") from e

    ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    train_all = ds["train"].filter(lambda ex: ex["task"] == TASK_ID)
    test_all = ds["test"].filter(lambda ex: ex["task"] == TASK_ID)

    labels = [int(x) for x in train_all["label"]]
    tr_idx, va_idx = _stratified_train_val_indices(labels, val_frac=args.val_frac, rng=rng)
    tr_idx = _cap_indices(tr_idx, args.train_max, rng)
    va_idx = _cap_indices(va_idx, args.val_max, rng)

    test_idx = list(range(len(test_all)))
    test_idx = _cap_indices(test_idx, args.test_max, rng)

    rows: List[Tuple[str, int, str]] = []
    for i in tr_idx:
        rows.append((train_all[i]["sequence"], int(train_all[i]["label"]), "train"))
    for i in va_idx:
        rows.append((train_all[i]["sequence"], int(train_all[i]["label"]), "val"))
    for i in test_idx:
        rows.append((test_all[i]["sequence"], int(test_all[i]["label"]), "test"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "label", "split"])
        w.writeheader()
        for seq, y, split_name in rows:
            w.writerow({"sequence": seq, "label": y, "split": split_name})

    n_pos = sum(1 for _, y, _ in rows if y == 1)
    by_split: Dict[str, int] = {}
    for _, _, s in rows:
        by_split[s] = by_split.get(s, 0) + 1
    print(f"[prepare_splice_data] wrote: {args.out}")
    print(f"[prepare_splice_data] rows={len(rows)} positives={n_pos} negatives={len(rows) - n_pos}")
    print(f"[prepare_splice_data] by_split={by_split} val_frac(of_train)={args.val_frac}")
    print("[prepare_splice_data] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
