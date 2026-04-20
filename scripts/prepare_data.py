#!/usr/bin/env python3
"""
Day 2 — Prepare a minimal dataset for a sequence-level binary task (e.g. promoter vs non-promoter).

What you are building toward
----------------------------
Downstream training (Day 3) needs a table of labeled sequences:

  sequence (DNA string)  |  label (0/1)

This script focuses on *data plumbing* and *reproducible splits*, not biology-perfect labels.

Key concepts
------------
**Supervised learning framing**
  You assume each training example is (x, y): input sequence x and target label y.
  The model will learn a mapping from x → y under your chosen loss.

**Train / validation / test split**
  *Train*: parameters update here.
  *Validation*: tune hyperparameters, early stopping, model selection.
  *Test*: final unbiased estimate — only touch once (ideally) after you lock decisions.

  If positives and negatives are processed differently, stratify splits so each
  split has similar class balance.

**Data leakage (conceptual)**
  Leakage means information from “the future” or from duplicated near-identical
  sequences appears in both train and test, inflating metrics. For genomics:
  watch for overlapping windows, duplicated contigs, or same gene in multiple splits.

**Synthetic demo data**
  By default this script writes a *tiny toy* CSV so you can run the pipeline
  without downloading large corpora. Replace with a real promoter dataset when ready.

**Real data sources (typical patterns)**
  - Hugging Face `datasets`: `load_dataset("...", name="task")` if a task is published.
  - Public FASTA + BED/CSV labels you preprocess yourself.

Outputs
-------
Writes CSV with columns: sequence, label, split
Default path: data/processed/promoters_demo.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/promoters_demo.csv"),
        help="Output CSV path (parent dirs are created).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible splits and synthetic sequence generation.",
    )
    p.add_argument(
        "--n_total",
        type=int,
        default=200,
        help="Number of synthetic rows to generate (small on purpose).",
    )
    return p.parse_args()


def _dna_rng(rng: random.Random, length: int) -> str:
    """Uniform random DNA string over {A,C,G,T}."""
    alphabet = "ACGT"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _inject_motif(seq: str, motif: str, rng: random.Random) -> str:
    """Insert motif at a random position (keeps length ~ same by trimming tail)."""
    if len(seq) < len(motif) + 1:
        return motif[: len(seq)]
    i = rng.randint(0, len(seq) - len(motif))
    return seq[:i] + motif + seq[i + len(motif) :]

# produces and 1/2 positive and 1/2 negative sequences with the motif in the positive sequences
# list of tuples with the sequence and the label(contains promoter or not)
def make_synthetic_rows(n_total: int, rng: random.Random) -> List[Tuple[str, int]]:
    """
    Build toy (sequence, label) pairs.

    Label rule (toy only):
      label=1 if a rough TATA-box-like substring appears; else 0.

    This is NOT a biologically valid promoter definition — it exists so you can
    test code paths (balance, splits, file IO) before swapping in real labels.
    """
    motif = "TATAAA"
    rows: List[Tuple[str, int]] = []
    # loop over for total number of sequences
    for _ in range(n_total):
        length = rng.randint(120, 240)  # random length between 120 and 240
        seq = _dna_rng(rng, length)     # generate a random DNA sequence of the length
        is_pos = rng.random() < 0.5     # random chance of being a positive sequence
        if is_pos:
            seq = _inject_motif(seq, motif, rng) # inject the motif into the sequence
            label = 1 if motif in seq else 0
        else:
            # Negative: resample if motif accidentally appears.
            while motif in seq:
                seq = _dna_rng(rng, length) # resample if the motif accidentally appears
            label = 0
        rows.append((seq, label))

        """produces and 1/2 positive and 1/2 negative sequences with the motif in the positive sequences"""
    return rows


def split_rows(
    rows: Iterable[Tuple[str, int]],
    rng: random.Random,
    frac_train: float = 0.7,
    frac_val: float = 0.15,
) -> List[Tuple[str, int, str]]:
    """
    Assign each row to train/val/test.

    Returns list of (sequence, label, split_name).
    """
    rows_list = list(rows)
    rng.shuffle(rows_list)

    n = len(rows_list)
    n_train = int(n * frac_train)
    n_val = int(n * frac_val)
    # remainder goes to test

    out: List[Tuple[str, int, str]] = []
    for i, (seq, y) in enumerate(rows_list):
        if i < n_train:
            split_name = "train"
        elif i < n_train + n_val:
            split_name = "val"
        else:
            split_name = "test"
        out.append((seq, y, split_name))
    return out


def write_csv(path: Path, records: List[Tuple[str, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "label", "split"])
        w.writeheader()
        for seq, y, split_name in records:
            w.writerow({"sequence": seq, "label": y, "split": split_name})


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    rows = make_synthetic_rows(args.n_total, rng)
    labeled = split_rows(rows, rng)
    write_csv(args.out, labeled)

    n_pos = sum(1 for _, y, _ in labeled if y == 1)
    print(f"[prepare_data] wrote: {args.out}")
    print(f"[prepare_data] rows={len(labeled)} positives={n_pos} negatives={len(labeled) - n_pos}")
    print("[prepare_data] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
