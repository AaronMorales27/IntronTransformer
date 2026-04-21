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
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_f = float(loss.item())
            total_loss += loss_f

            # Heartbeat: first epoch batch is slow on MPS/CUDA cold start; no prints looked like a hang.
            if batch_idx == 0:
                print(
                    f"[train] epoch={epoch} first batch ok | {n_batches} batches/epoch | running...",
                    flush=True,
                )
            elif (batch_idx + 1) % 300 == 0:
                print(
                    f"[train] epoch={epoch} batch={batch_idx + 1}/{n_batches} loss={loss_f:.4f}",
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device=device)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(
            f"[train] epoch={epoch} loss={avg_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

