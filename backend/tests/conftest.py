"""Shared fixtures. Production code never uses these mock weights."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.config import get_settings
from app.ml.architecture import build_classifier
from app.ml.metadata import ModelMetadata, save_metadata
from app.main import create_app


@pytest.fixture
def rgb_png_bytes() -> bytes:
    image = Image.new("RGB", (96, 96), color=(40, 80, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    metadata = ModelMetadata(model_version="test_fixture_not_for_production")
    model = build_classifier("efficientnet_b0", pretrained_backbone=False)
    ckpt = tmp_path / "deepshield_efficientnet_b0.pt"
    meta = tmp_path / "model_metadata.json"
    torch.save({"state_dict": model.state_dict(), "metadata": metadata.to_json()}, ckpt)
    save_metadata(meta, metadata)
    return ckpt, meta


@pytest.fixture
def client_without_model(monkeypatch, tmp_path: Path) -> TestClient:
    get_settings.cache_clear()
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


@pytest.fixture
def client_with_fixture_model(monkeypatch, tiny_checkpoint) -> TestClient:
    ckpt, meta = tiny_checkpoint
    get_settings.cache_clear()
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("MODEL_DIR", str(ckpt.parent))
    monkeypatch.setenv("CHECKPOINT_FILENAME", ckpt.name)
    monkeypatch.setenv("METADATA_FILENAME", meta.name)
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()
