"""Checkpoint loading and compatibility checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.ml.architecture import build_classifier
from app.ml.metadata import ModelMetadata, load_metadata


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be used for production inference."""


def assert_state_dict_matches(architecture: str, state_dict: dict[str, Any]) -> None:
    """Reject a checkpoint whose tensors do not belong to the named architecture.

    The filename is ignored. A file named efficientnet can still hold artifact_cnn
    weights if that is what training saved and metadata records.
    """
    if architecture == "artifact_cnn":
        weight = state_dict.get("classifier.2.weight")
        if weight is None or tuple(weight.shape) != (2, 192):
            raise CheckpointError(
                "Metadata says artifact_cnn, but these weights do not match that network."
            )
        return
    if architecture == "efficientnet_b0":
        weight = state_dict.get("classifier.1.weight")
        if weight is None or int(weight.shape[0]) != 2:
            raise CheckpointError(
                "Metadata says efficientnet_b0, but these weights do not match that network."
            )
        return
    raise CheckpointError(f"Unsupported architecture '{architecture}'.")


def load_checkpoint_bundle(
    checkpoint_path: Path,
    metadata_path: Path,
    device: torch.device,
) -> tuple[nn.Module, ModelMetadata, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise CheckpointError(
            f"No checkpoint found at {checkpoint_path}. Train a model and copy the .pt file here."
        )
    try:
        bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - surface load failures as checkpoint errors
        raise CheckpointError("Checkpoint file is unreadable or incompatible with this PyTorch version.") from exc

    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise CheckpointError(
            "Checkpoint must be a dict containing 'state_dict'. Re-export using training/train.py."
        )
    metadata = load_metadata(metadata_path)
    if metadata is None and "metadata" in bundle:
        metadata = ModelMetadata.from_dict(bundle["metadata"])
    if metadata is None:
        raise CheckpointError("Missing model_metadata.json and no metadata inside the checkpoint.")
    if metadata.model_version in {"", "untrained"}:
        raise CheckpointError(
            "Checkpoint metadata still reports version 'untrained'. Only trained evaluation-selected checkpoints are served."
        )
    assert_state_dict_matches(metadata.architecture, bundle["state_dict"])
    model = build_classifier(metadata.architecture, pretrained_backbone=False)
    try:
        model.load_state_dict(bundle["state_dict"])
    except RuntimeError as exc:
        raise CheckpointError(
            f"Checkpoint weights do not match architecture '{metadata.architecture}'."
        ) from exc
    model.to(device)
    model.eval()
    extras = {k: v for k, v in bundle.items() if k not in {"state_dict", "metadata"}}
    return model, metadata, extras
