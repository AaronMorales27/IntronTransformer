from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader


def collect_labels_and_probs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[List[int], List[float]]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)[:, 1]

            y_true.extend(labels.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
    return y_true, y_prob


def metrics_from_labels_probs(y_true: List[int], y_prob: List[float], threshold: float) -> Dict[str, float]:
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true, y_prob = collect_labels_and_probs(model=model, loader=loader, device=device)
    return metrics_from_labels_probs(y_true=y_true, y_prob=y_prob, threshold=threshold)


def select_threshold_max_f1(y_true: List[int], y_prob: List[float]) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.arange(0.05, 0.951, 0.01):
        f1 = metrics_from_labels_probs(y_true, y_prob, float(t))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t

