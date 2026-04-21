from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# On some macOS setups, duplicate OpenMP libraries can crash PyTorch.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import AutoTokenizer

from src.pipeline.config import ExperimentConfig
from src.pipeline.data import build_dataloaders, read_split_rows, validate_splits
from src.pipeline.evaluate import collect_labels_and_probs, evaluate, select_threshold_max_f1
from src.pipeline.modeling import (
    FrozenBackboneClassifier,
    count_trainable_params,
    make_backbone,
    set_trainable_params,
)
from src.pipeline.train import train_head_only


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a frozen-backbone classifier and compare pretrained vs random initialization."
    )
    p.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    p.add_argument("--data_csv", type=Path, default=None)
    p.add_argument("--model_id", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device. auto picks cuda > mps > cpu.",
    )
    p.add_argument("--out_csv", type=Path, default=None)
    p.add_argument(
        "--tune_mode",
        type=str,
        default=None,
        choices=["frozen_head_only", "partial_unfreeze"],
        help="Training mode: head-only or unfreeze top N backbone blocks.",
    )
    p.add_argument(
        "--unfreeze_top_n",
        type=int,
        default=None,
        help="For partial_unfreeze mode, number of top transformer blocks to unfreeze.",
    )
    p.add_argument(
        "--ladder_top_n",
        type=str,
        default=None,
        help="Comma-separated unfreeze ladder, e.g. '1,2,4'. Runs frozen + each top-N setting.",
    )
    p.add_argument(
        "--include_random_anchor",
        action="store_true",
        help="Also run random-initialized anchor for the selected mode(s).",
    )
    p.add_argument(
        "--sweep_thresholds",
        action="store_true",
        help="Pick threshold from validation split by maximizing F1.",
    )
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig.defaults()
    if args.config is not None:
        cfg = ExperimentConfig.from_yaml(args.config)
    return cfg.with_overrides(
        {
            "data_csv": args.data_csv,
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": args.device,
            "out_csv": args.out_csv,
            "sweep_thresholds": True if args.sweep_thresholds else None,
            "tune_mode": args.tune_mode,
            "unfreeze_top_n": args.unfreeze_top_n,
            "ladder_top_n": args.ladder_top_n,
            "include_random_anchor": True if args.include_random_anchor else None,
        }
    )


def parse_ladder(raw: str) -> List[int]:
    vals: List[int] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        n = int(s)
        if n <= 0:
            raise ValueError(f"ladder_top_n entries must be > 0, got {n}")
        vals.append(n)
    dedup = sorted(set(vals))
    if not dedup:
        raise ValueError("ladder_top_n is empty; expected values like '1,2,4'")
    return dedup


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_metrics(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "split",
        "accuracy",
        "f1",
        "roc_auc",
        "epochs",
        "batch_size",
        "lr",
        "max_length",
        "seed",
        "selected_threshold",
        "threshold_source",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_experiment(
    run_name: str,
    cfg: ExperimentConfig,
    loaders,
    device: str,
) -> List[Dict[str, object]]:
    pretrained = run_name.startswith("pretrained")
    backbone, hidden_size = make_backbone(model_id=cfg.model_id, pretrained=pretrained)
    model = FrozenBackboneClassifier(backbone=backbone, hidden_size=hidden_size, num_classes=2)
    unfrozen_layers = set_trainable_params(
        model=model,
        tune_mode=cfg.tune_mode,
        unfreeze_top_n=cfg.unfreeze_top_n,
    )
    trainable = count_trainable_params(model)
    print(
        f"[run] {run_name} mode={cfg.tune_mode} "
        f"unfreeze_top_n={cfg.unfreeze_top_n} unfrozen_layers={unfrozen_layers} "
        f"trainable_params={trainable}"
    )

    train_head_only(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )

    if cfg.sweep_thresholds:
        val_true, val_prob = collect_labels_and_probs(model=model, loader=loaders["val"], device=device)
        selected_threshold = select_threshold_max_f1(y_true=val_true, y_prob=val_prob)
        threshold_source = "val_f1_sweep"
    else:
        selected_threshold = 0.5
        threshold_source = "fixed_0.5"

    print(f"[run] {run_name} threshold={selected_threshold:.2f} source={threshold_source}")

    rows: List[Dict[str, object]] = []
    for split_name in ("val", "test"):
        metrics = evaluate(model, loaders[split_name], device, threshold=selected_threshold)
        rows.append(
            {
                "run_name": run_name,
                "split": split_name,
                "accuracy": f"{metrics['accuracy']:.6f}",
                "f1": f"{metrics['f1']:.6f}",
                "roc_auc": f"{metrics['roc_auc']:.6f}",
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "max_length": cfg.max_length,
                "seed": cfg.seed,
                "selected_threshold": f"{selected_threshold:.2f}",
                "threshold_source": threshold_source,
                "notes": (
                    f"tune_mode={cfg.tune_mode}; unfreeze_top_n={cfg.unfreeze_top_n}; "
                    "mean-pooled linear classifier"
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    cfg = resolve_config(args)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    if not cfg.data_csv.exists():
        raise FileNotFoundError(f"Data CSV not found: {cfg.data_csv}")

    splits = read_split_rows(cfg.data_csv)
    validate_splits(splits, source=cfg.data_csv)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    loaders = build_dataloaders(
        splits=splits,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
    )

    print(f"[setup] device={device}")
    print(
        f"[setup] rows train/val/test = "
        f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}"
    )

    all_rows: List[Dict[str, object]] = []
    if cfg.ladder_top_n.strip():
        ladder = parse_ladder(cfg.ladder_top_n)
        base_cfg = cfg.with_overrides({"tune_mode": "frozen_head_only", "unfreeze_top_n": 0})
        all_rows.extend(run_experiment("pretrained_frozen_head_only", cfg=base_cfg, loaders=loaders, device=device))
        for n in ladder:
            step_cfg = cfg.with_overrides({"tune_mode": "partial_unfreeze", "unfreeze_top_n": n})
            all_rows.extend(
                run_experiment(f"pretrained_partial_unfreeze_top{n}", cfg=step_cfg, loaders=loaders, device=device)
            )
        if cfg.include_random_anchor:
            all_rows.extend(run_experiment("random_frozen_head_only", cfg=base_cfg, loaders=loaders, device=device))
    elif cfg.tune_mode == "partial_unfreeze" and cfg.unfreeze_top_n > 0:
        all_rows.extend(
            run_experiment(
                f"pretrained_partial_unfreeze_top{cfg.unfreeze_top_n}",
                cfg=cfg,
                loaders=loaders,
                device=device,
            )
        )
        if cfg.include_random_anchor:
            all_rows.extend(
                run_experiment(
                    f"random_partial_unfreeze_top{cfg.unfreeze_top_n}",
                    cfg=cfg,
                    loaders=loaders,
                    device=device,
                )
            )
    else:
        all_rows.extend(run_experiment("pretrained_frozen_head_only", cfg=cfg, loaders=loaders, device=device))
        if cfg.include_random_anchor:
            all_rows.extend(run_experiment("random_frozen_head_only", cfg=cfg, loaders=loaders, device=device))

    write_metrics(cfg.out_csv, all_rows)
    print(f"[done] wrote metrics: {cfg.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

