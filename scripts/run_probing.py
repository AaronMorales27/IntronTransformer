from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoTokenizer

# Keep script execution behavior consistent with scripts/train_classifier.py.
# This makes `python3 scripts/run_probing.py ...` robust across local envs.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import ExperimentConfig
from src.pipeline.data import build_dataloaders, read_split_rows, validate_splits
from src.pipeline.modeling import FrozenBackboneClassifier, make_backbone


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a layer-wise probing experiment on frozen transformer embeddings. "
            "A linear probe is trained independently for each layer."
        )
    )
    p.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    p.add_argument("--data_csv", type=Path, default=None)
    p.add_argument("--model_id", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device. auto picks cuda > mps > cpu.",
    )
    p.add_argument("--out_csv", type=Path, default=Path("results/probe_scores.csv"))
    p.add_argument("--out_png", type=Path, default=Path("results/fig_probe_layerwise.png"))
    p.add_argument(
        "--threshold_mode",
        type=str,
        default="val_sweep",
        choices=["fixed", "val_sweep"],
        help=(
            "How to pick classification threshold per layer: "
            "'fixed' uses 0.5, 'val_sweep' picks best threshold on validation F1."
        ),
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Comma-separated seeds for multi-seed probing, e.g. '42,43,44'. "
            "If omitted, uses --seed / config seed."
        ),
    )
    p.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "accuracy", "roc_auc"],
        help="Metric used in the plot and printed highlights.",
    )
    return p.parse_args()


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


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig.defaults()
    if args.config is not None:
        cfg = ExperimentConfig.from_yaml(args.config)
    return cfg.with_overrides(
        {
            "data_csv": args.data_csv,
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": args.device,
        }
    )


def _forward_with_hidden_states(backbone, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Handle checkpoint API differences (some expect encoder_attention_mask)."""
    try:
        return backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )
    except TypeError:
        return backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )


def _masked_mean(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool token embeddings into one sequence vector, ignoring padding tokens."""
    return FrozenBackboneClassifier.masked_mean(hidden_state, attention_mask)


def extract_layerwise_features(
    backbone,
    loader,
    device: str,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Convert each sequence into one vector per layer.

    Why this shape:
    - We probe each layer separately, so we keep one design matrix X[layer_idx].
    - Each row in X[layer_idx] is one sequence.
    """
    backbone.eval()
    layer_buckets: List[List[np.ndarray]] = []
    labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].cpu().numpy()

            out = _forward_with_hidden_states(backbone, input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = out.hidden_states

            if not layer_buckets:
                layer_buckets = [[] for _ in range(len(hidden_states))]

            for layer_idx, layer_h in enumerate(hidden_states):
                pooled = _masked_mean(layer_h, attention_mask).cpu().numpy()
                layer_buckets[layer_idx].append(pooled)
            labels.append(y)

    x_by_layer = [np.concatenate(parts, axis=0) for parts in layer_buckets]
    y_all = np.concatenate(labels, axis=0)
    return x_by_layer, y_all


def score_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def pick_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Pick threshold by maximizing validation F1.

    Using validation split avoids leaking test labels while letting each layer
    choose a threshold that matches its probability calibration.
    """
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = score_binary(y_true=y_true, y_prob=y_prob, threshold=float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold


def parse_seed_list(args: argparse.Namespace, cfg: ExperimentConfig) -> List[int]:
    if args.seeds is None:
        return [int(cfg.seed)]
    parts = [x.strip() for x in args.seeds.split(",") if x.strip()]
    if not parts:
        raise ValueError("`--seeds` was provided but no valid integers were found.")
    return [int(x) for x in parts]


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> LogisticRegression:
    """
    Fit a lightweight linear classifier as a probe.

    Design decision:
    - Keep probe capacity small (logistic regression) so scores mostly reflect
      the representation quality of the layer, not a powerful downstream model.
    """
    probe = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="liblinear",
    )
    probe.fit(x_train, y_train)
    return probe


def write_probe_scores(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer_idx",
        "split",
        "seed",
        "threshold_mode",
        "selected_threshold",
        "accuracy",
        "f1",
        "roc_auc",
        "num_examples",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_plot(path: Path, rows: Sequence[Dict[str, object]], metric: str) -> None:
    """
    Optional plotting: if matplotlib is unavailable, keep CSV output as source of truth.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib unavailable; skipping figure generation.")
        return

    mean_rows = [r for r in rows if r["split"] == "test_mean"]
    if mean_rows:
        xs = [int(r["layer_idx"]) for r in mean_rows]
        ys = [float(r[metric]) for r in mean_rows]
    else:
        test_rows = [r for r in rows if r["split"] == "test"]
        xs = [int(r["layer_idx"]) for r in test_rows]
        ys = [float(r[metric]) for r in test_rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Layer index")
    plt.ylabel(metric.upper())
    plt.title(f"Layer-wise probe performance ({metric})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()
    cfg = resolve_config(args)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    seeds = parse_seed_list(args, cfg)

    if not cfg.data_csv.exists():
        raise FileNotFoundError(f"Data CSV not found: {cfg.data_csv}")

    print(f"[setup] device={device}")
    print(f"[setup] model_id={cfg.model_id}")
    print(f"[setup] data_csv={cfg.data_csv}")
    print(f"[setup] threshold_mode={args.threshold_mode}")
    print(f"[setup] seeds={seeds}")

    splits = read_split_rows(cfg.data_csv)
    validate_splits(splits, source=cfg.data_csv)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    loaders = build_dataloaders(
        splits=splits,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
    )

    backbone, _ = make_backbone(model_id=cfg.model_id, pretrained=True)
    backbone = backbone.to(device)

    print("[step] extracting train layer embeddings")
    x_train_by_layer, y_train = extract_layerwise_features(backbone=backbone, loader=loaders["train"], device=device)
    print("[step] extracting val layer embeddings")
    x_val_by_layer, y_val = extract_layerwise_features(backbone=backbone, loader=loaders["val"], device=device)
    print("[step] extracting test layer embeddings")
    x_test_by_layer, y_test = extract_layerwise_features(backbone=backbone, loader=loaders["test"], device=device)

    if not (len(x_train_by_layer) == len(x_val_by_layer) == len(x_test_by_layer)):
        raise RuntimeError("Mismatch in number of extracted layers between train/val/test.")

    rows: List[Dict[str, object]] = []
    num_layers = len(x_train_by_layer)
    per_layer_seed_metrics: Dict[int, Dict[str, List[float]]] = {
        layer_idx: {"accuracy": [], "f1": [], "roc_auc": []} for layer_idx in range(num_layers)
    }

    for seed in seeds:
        print(f"[seed] running probe seed={seed}")
        for layer_idx in range(num_layers):
            probe = fit_probe(
                x_train=x_train_by_layer[layer_idx],
                y_train=y_train,
                seed=seed,
            )
            y_val_prob = probe.predict_proba(x_val_by_layer[layer_idx])[:, 1]
            y_test_prob = probe.predict_proba(x_test_by_layer[layer_idx])[:, 1]

            if args.threshold_mode == "val_sweep":
                selected_threshold = pick_best_threshold(y_true=y_val, y_prob=y_val_prob)
            else:
                selected_threshold = 0.5

            metrics = score_binary(y_true=y_test, y_prob=y_test_prob, threshold=selected_threshold)
            per_layer_seed_metrics[layer_idx]["accuracy"].append(metrics["accuracy"])
            per_layer_seed_metrics[layer_idx]["f1"].append(metrics["f1"])
            per_layer_seed_metrics[layer_idx]["roc_auc"].append(metrics["roc_auc"])

            print(
                f"[probe] seed={seed} layer={layer_idx:02d} thr={selected_threshold:.2f} "
                f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f}"
            )
            rows.append(
                {
                    "layer_idx": layer_idx,
                    "split": "test",
                    "seed": seed,
                    "threshold_mode": args.threshold_mode,
                    "selected_threshold": f"{selected_threshold:.2f}",
                    "accuracy": f"{metrics['accuracy']:.6f}",
                    "f1": f"{metrics['f1']:.6f}",
                    "roc_auc": f"{metrics['roc_auc']:.6f}",
                    "num_examples": len(y_test),
                    "notes": "pretrained frozen backbone + logistic regression probe (per-seed)",
                }
            )

    for layer_idx in range(num_layers):
        stats = per_layer_seed_metrics[layer_idx]
        acc_mean = float(np.mean(stats["accuracy"]))
        f1_mean = float(np.mean(stats["f1"]))
        auc_mean = float(np.mean(stats["roc_auc"]))
        acc_std = float(np.std(stats["accuracy"], ddof=0))
        f1_std = float(np.std(stats["f1"], ddof=0))
        auc_std = float(np.std(stats["roc_auc"], ddof=0))
        rows.append(
            {
                "layer_idx": layer_idx,
                "split": "test_mean",
                "seed": "all",
                "threshold_mode": args.threshold_mode,
                "selected_threshold": "varies_per_seed_layer",
                "accuracy": f"{acc_mean:.6f}",
                "f1": f"{f1_mean:.6f}",
                "roc_auc": f"{auc_mean:.6f}",
                "num_examples": len(y_test),
                "notes": (
                    f"mean across {len(seeds)} seeds; "
                    f"std(acc/f1/auc)=({acc_std:.4f}/{f1_std:.4f}/{auc_std:.4f})"
                ),
            }
        )

    summary_rows = [r for r in rows if r["split"] == "test_mean"] or [r for r in rows if r["split"] == "test"]
    best_row = max(summary_rows, key=lambda r: float(r[args.metric]))
    print(
        f"[result] best layer by {args.metric}: "
        f"{best_row['layer_idx']} ({float(best_row[args.metric]):.4f})"
    )

    write_probe_scores(args.out_csv, rows)
    maybe_plot(args.out_png, rows, metric=args.metric)
    print(f"[done] wrote probe scores: {args.out_csv}")
    print(f"[done] wrote probe figure: {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
