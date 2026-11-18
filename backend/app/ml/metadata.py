"""Model metadata and versioning helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ModelMetadata:
    model_name: str = "deepshield_efficientnet_b0"
    model_version: str = "untrained"
    architecture: str = "efficientnet_b0"
    classes: list[str] = field(default_factory=lambda: ["authentic", "manipulated"])
    authentic_index: int = 0
    manipulated_index: int = 1
    input_size: int = 224
    image_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    authentic_threshold: float = 0.35
    manipulated_threshold: float = 0.65
    score_type: str = "model_score"
    requires_face_crop: bool = True
    dataset_name: str | None = None
    dataset_source: str | None = None
    dataset_license: str | None = None
    split_strategy: str | None = None
    random_seed: int | None = None
    trained_at: str | None = None
    hardware: str | None = None
    metrics: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def load_metadata(path: Path) -> ModelMetadata | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return ModelMetadata.from_dict(json.load(handle))


def save_metadata(path: Path, metadata: ModelMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_json(), handle, indent=2)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
