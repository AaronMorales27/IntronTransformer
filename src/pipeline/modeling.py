from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM


class FrozenBackboneClassifier(nn.Module):
    """
    Frozen backbone + trainable head.

    Geometry intuition:
    - backbone maps sequence tokens to a cloud of token vectors in high-D space.
    - mean pooling compresses token cloud -> one sequence vector.
    - linear head learns a separating plane in that fixed space.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(hidden_size, num_classes)

    @staticmethod
    def masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        masked = last_hidden_state * mask
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return masked.sum(dim=1) / lengths

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Some genomic checkpoints expect encoder_attention_mask.
        try:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True,
            )
        except TypeError:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Use final transformer hidden states as sequence features.
        last_hidden_state = out.hidden_states[-1]
        seq_embedding = self.masked_mean(last_hidden_state, attention_mask)
        return self.classifier(seq_embedding)


def make_backbone(model_id: str, pretrained: bool) -> Tuple[nn.Module, int]:
    if pretrained:
        backbone = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
        config = backbone.config
    else:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        backbone = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "d_model", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size from model config.")
    return backbone, int(hidden_size)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

