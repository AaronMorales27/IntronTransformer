from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.pipeline.evaluate import evaluate
from src.pipeline.modeling import FrozenBackboneClassifier


def train_head_only(
    model: FrozenBackboneClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
) -> None:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_metrics = evaluate(model, val_loader, device=device)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(
            f"[train] epoch={epoch} loss={avg_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

