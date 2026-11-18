"""Process-wide model holder. Loaded once at startup."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from app.config import Settings
from app.ml.architecture import resolve_device
from app.ml.checkpoint import CheckpointError, load_checkpoint_bundle
from app.ml.faces import FaceDetector
from app.ml.inference import ImageDetector
from app.ml.metadata import ModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class ModelRuntime:
    loaded: bool
    checkpoint_present: bool
    reason: str | None
    model: nn.Module | None = None
    metadata: ModelMetadata | None = None
    detector: ImageDetector | None = None
    device: torch.device | None = None
    semaphore: torch.cuda.Stream | None = None


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.runtime = ModelRuntime(loaded=False, checkpoint_present=False, reason="not_initialized")
        self._lock = None

    def load(self) -> None:
        import threading

        self._lock = threading.Semaphore(self.settings.inference_max_concurrency)
        device = resolve_device(self.settings.device)
        checkpoint_present = self.settings.checkpoint_path.exists()
        try:
            model, metadata, _extras = load_checkpoint_bundle(
                self.settings.checkpoint_path,
                self.settings.metadata_path,
                device,
            )
            detector = ImageDetector(model, metadata, device, FaceDetector())
            self.runtime = ModelRuntime(
                loaded=True,
                checkpoint_present=True,
                reason=None,
                model=model,
                metadata=metadata,
                detector=detector,
                device=device,
            )
            logger.info("Loaded checkpoint %s", self.settings.checkpoint_path)
        except CheckpointError as exc:
            self.runtime = ModelRuntime(
                loaded=False,
                checkpoint_present=checkpoint_present,
                reason=str(exc),
                device=device,
            )
            logger.warning("Model not ready: %s", exc)

    def require_detector(self) -> ImageDetector:
        if not self.runtime.loaded or self.runtime.detector is None or self.runtime.metadata is None:
            raise RuntimeError(self.runtime.reason or "Model is not loaded.")
        return self.runtime.detector
