"""Image upload inference."""

from __future__ import annotations

import time

from fastapi import HTTPException, status

from app.config import Settings
from app.ml.preprocess import MediaError, decode_image_bytes
from app.schemas.detection import DetectionResponse
from app.services.model_service import ModelService
from app.services.score_service import image_response, unavailable_image


class ImageService:
    def __init__(self, settings: Settings, model_service: ModelService) -> None:
        self.settings = settings
        self.model_service = model_service

    def analyze_bytes(self, data: bytes, subject: str = "image") -> DetectionResponse:
        start = time.perf_counter()
        if not self.model_service.runtime.loaded:
            return unavailable_image(self.model_service.runtime.reason or "Checkpoint not installed.")
        try:
            image = decode_image_bytes(data, self.settings.max_image_pixels, self.settings.min_image_side)
        except MediaError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        detector = self.model_service.require_detector()
        assert self.model_service._lock is not None
        with self.model_service._lock:
            result = detector.analyze(image, subject=subject)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        assert self.model_service.runtime.metadata is not None
        return image_response(result, self.model_service.runtime.metadata, elapsed_ms)
