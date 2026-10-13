"""Typed configuration loaded from YAML.

Unknown keys raise an error so that a typo in a config file cannot silently fall back to a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

SAMPLING_METHODS = ("kcenter_greedy", "random")


@dataclass(frozen=True)
class ModelConfig:
    name: str = "patchcore"
    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    pre_trained: bool = True
    coreset_ratio: float = 0.1
    num_neighbors: int = 9
    sampling: str = "kcenter_greedy"  # memory-bank subsampling: "kcenter_greedy" | "random"
    sampling_seed: int = 0  # fixes the k-center start point / random projection / random subset


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int = 256
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataConfig:
    val_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class ThresholdConfig:
    method: str = "val_normal_quantile"
    image_quantile: float = 0.99
    pixel_quantile: float = 0.999


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    batch_size: int = 8
    num_threads: int = 4


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        if not 0 < self.model.coreset_ratio <= 1:
            raise ValueError(f"model.coreset_ratio must be in (0, 1], got {self.model.coreset_ratio}")
        if self.model.sampling not in SAMPLING_METHODS:
            raise ValueError(f"model.sampling must be one of {SAMPLING_METHODS}, got {self.model.sampling}")
        if not 0 < self.data.val_fraction < 1:
            raise ValueError(f"data.val_fraction must be in (0, 1), got {self.data.val_fraction}")
        for name in ("image_quantile", "pixel_quantile"):
            q = getattr(self.threshold, name)
            if not 0 < q <= 1:
                raise ValueError(f"threshold.{name} must be in (0, 1], got {q}")
        if self.threshold.method != "val_normal_quantile":
            raise ValueError(f"Unsupported threshold.method: {self.threshold.method}")
        if self.preprocessing.image_size < 32:
            raise ValueError("preprocessing.image_size must be >= 32")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "model": ModelConfig,
    "preprocessing": PreprocessingConfig,
    "data": DataConfig,
    "threshold": ThresholdConfig,
    "runtime": RuntimeConfig,
}


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in [{section}]: {sorted(unknown)}")
    # YAML lists become tuples so the config stays hashable and immutable.
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})


def config_from_dict(raw: dict[str, Any]) -> Config:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    return Config(**{name: _build(cls, raw.get(name) or {}, name) for name, cls in _SECTIONS.items()})


def load_config(path: str | Path, overrides: dict[str, dict[str, Any]] | None = None) -> Config:
    """Load a YAML config, optionally applying ``{"section": {"key": value}}`` overrides."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section, values in (overrides or {}).items():
        raw.setdefault(section, {}).update(values)
    return config_from_dict(raw)
