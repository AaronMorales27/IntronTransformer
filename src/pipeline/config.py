from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when dependency missing
    yaml = None


@dataclass(frozen=True)
class ExperimentConfig:
    data_csv: Path = Path("data/processed/promoters_demo.csv")
    model_id: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    batch_size: int = 8
    epochs: int = 3
    lr: float = 1e-3
    max_length: int = 256
    seed: int = 42
    device: str = "auto"
    out_csv: Path = Path("results/metrics_baseline_vs_pretrained.csv")
    sweep_thresholds: bool = False # 0.5 is the default threshold

    @staticmethod
    def defaults() -> "ExperimentConfig":
        return ExperimentConfig()

    @staticmethod
    def from_yaml(path: Path) -> "ExperimentConfig":
        if yaml is None:
            raise ImportError(
                "PyYAML is required for --config support. Install with: "
                "python -m pip install pyyaml"
            )
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config file must contain a YAML mapping at top level: {path}")
        return ExperimentConfig.from_dict(raw)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "ExperimentConfig":
        defaults = ExperimentConfig.defaults()
        allowed_keys = set(defaults.__dict__.keys())
        unknown = set(raw.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")

        merged = {**defaults.__dict__, **raw}
        return ExperimentConfig(
            data_csv=Path(merged["data_csv"]),
            model_id=str(merged["model_id"]),
            batch_size=int(merged["batch_size"]),
            epochs=int(merged["epochs"]),
            lr=float(merged["lr"]),
            max_length=int(merged["max_length"]),
            seed=int(merged["seed"]),
            device=str(merged["device"]),
            out_csv=Path(merged["out_csv"]),
            sweep_thresholds=bool(merged["sweep_thresholds"]),
        )

    def with_overrides(self, overrides: Dict[str, Any]) -> "ExperimentConfig":
        clean = {k: v for k, v in overrides.items() if v is not None}
        merged = {**self.__dict__, **clean}
        return ExperimentConfig.from_dict(merged)

