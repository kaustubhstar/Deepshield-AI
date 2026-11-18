"""Load the installed checkpoint and confirm it is the network that was trained."""

from pathlib import Path

import pytest
import torch

from app.ml.architecture import build_classifier
from app.ml.checkpoint import CheckpointError, assert_state_dict_matches, load_checkpoint_bundle

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "deepshield_efficientnet_b0.pt"
METADATA = ROOT / "models" / "model_metadata.json"


def test_artifact_and_efficientnet_weights_are_not_interchangeable():
    artifact = build_classifier("artifact_cnn").state_dict()
    efficient = build_classifier("efficientnet_b0", pretrained_backbone=False).state_dict()
    assert_state_dict_matches("artifact_cnn", artifact)
    assert_state_dict_matches("efficientnet_b0", efficient)
    with pytest.raises(CheckpointError):
        assert_state_dict_matches("efficientnet_b0", artifact)
    with pytest.raises(CheckpointError):
        assert_state_dict_matches("artifact_cnn", efficient)


def test_installed_checkpoint_loads_as_artifact_cnn():
    if not CHECKPOINT.exists() or not METADATA.exists():
        pytest.skip("Installed checkpoint is not in this workspace.")
    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert bundle["metadata"]["architecture"] == "artifact_cnn"
    assert_state_dict_matches("artifact_cnn", bundle["state_dict"])
    model, metadata, _extras = load_checkpoint_bundle(CHECKPOINT, METADATA, torch.device("cpu"))
    assert metadata.architecture == "artifact_cnn"
    assert metadata.input_size == 128
    assert metadata.authentic_index == 0
    assert metadata.manipulated_index == 1
    assert metadata.model_version != "untrained"
    assert tuple(model.classifier[2].weight.shape) == (2, 192)


def test_installed_model_image_result_uses_its_own_scores():
    if not CHECKPOINT.exists() or not METADATA.exists():
        pytest.skip("Installed checkpoint is not in this workspace.")
    from PIL import Image

    from app.ml.inference import ImageDetector

    model, metadata, _extras = load_checkpoint_bundle(CHECKPOINT, METADATA, torch.device("cpu"))
    detector = ImageDetector(model, metadata, torch.device("cpu"))
    result = detector.analyze(Image.new("RGB", (160, 160), color=(30, 90, 140)))
    assert result.verdict in {"REAL", "FAKE"}
    assert result.explanation
    assert 50.0 <= result.confidence_percent <= 100.0
    if result.verdict == "FAKE":
        assert result.score >= 0.5
        assert result.confidence_percent == round(result.score * 100, 1)
    else:
        assert result.score < 0.5
        assert result.confidence_percent == round((1 - result.score) * 100, 1)
