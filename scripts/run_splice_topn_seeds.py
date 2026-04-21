#!/usr/bin/env python3
"""
Run splice partial-unfreeze experiments for one top-N setting across many seeds.

This script shells out to scripts/train_classifier.py, writes one CSV per seed,
and produces a combined CSV for quick comparison.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/splice.yaml"),
        help="Base config passed to train_classifier.",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=4,
        help="Number of top transformer blocks to unfreeze.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds to run, e.g. '42,43,44'.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results"),
        help="Directory for per-seed and combined CSV outputs.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="splice_partial_unfreeze_top",
        help="Output filename prefix.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional batch size override passed to train_classifier.",
    )
    return p.parse_args()


def parse_seeds(raw: str) -> List[int]:
    out: List[int] = []
    for token in raw.split(","):
        s = token.strip()
        if not s:
            continue
        out.append(int(s))
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError("No seeds parsed from --seeds")
    return uniq


def run_one_seed(
    config: Path,
    top_n: int,
    seed: int,
    out_csv: Path,
    batch_size: int | None,
) -> None:
    cmd = [
        sys.executable,
        "scripts/train_classifier.py",
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--out_csv",
        str(out_csv),
        "--tune_mode",
        "partial_unfreeze",
        "--unfreeze_top_n",
        str(top_n),
    ]
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    print(f"[grid] running seed={seed} top_n={top_n} -> {out_csv}")
    subprocess.run(cmd, check=True)


def combine_csvs(csv_paths: List[Path], combined_out: Path) -> None:
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] | None = None
    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(r.fieldnames or [])
            for row in r:
                rows.append(row)
    if fieldnames is None:
        raise ValueError("No CSV rows found to combine.")
    combined_out.parent.mkdir(parents=True, exist_ok=True)
    with combined_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_paths: List[Path] = []
    for seed in seeds:
        out_csv = args.out_dir / f"{args.prefix}{args.top_n}_seed_{seed}.csv"
        run_one_seed(
            config=args.config,
            top_n=args.top_n,
            seed=seed,
            out_csv=out_csv,
            batch_size=args.batch_size,
        )
        per_seed_paths.append(out_csv)

    combined_out = args.out_dir / f"{args.prefix}{args.top_n}_combined.csv"
    combine_csvs(per_seed_paths, combined_out)
    print(f"[grid] combined rows written: {combined_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
