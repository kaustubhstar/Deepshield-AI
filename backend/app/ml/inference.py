"""Single-image inference using a loaded classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torch import nn

from app.ml.faces import FaceDetector
from app.ml.metadata import ModelMetadata
from app.ml.preprocess import build_transform
from app.ml.scores import decision_from_logits
from app.schemas.detection import PredictionLabel


@dataclass
class ImageInferenceResult:
    prediction: PredictionLabel
    score: float
    face_detected: bool
    warnings: list[str]
    verdict: str
    confidence_percent: float
    explanation: str


class ImageDetector:
    def __init__(
        self,
        model: nn.Module,
        metadata: ModelMetadata,
        device: torch.device,
        face_detector: FaceDetector | None = None,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.device = device
        self.face_detector = face_detector or FaceDetector()
        self.transform = build_transform(metadata)

    @torch.inference_mode()
    def analyze(self, image: Image.Image, subject: str = "image") -> ImageInferenceResult:
        warnings: list[str] = []
        working = image
        face_detected = False
        if self.metadata.requires_face_crop:
            working, face_detected = self.face_detector.crop(image)
            if not face_detected:
                warnings.append("No face was found, so the whole picture was checked. That result is less reliable.")
        tensor = self.transform(working).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        decision = decision_from_logits(logits, self.metadata, subject)
        return ImageInferenceResult(
            prediction=decision.prediction,
            score=decision.manipulated_score,
            face_detected=face_detected,
            warnings=warnings,
            verdict=decision.verdict,
            confidence_percent=decision.confidence_percent,
            explanation=decision.explanation,
        )
