"""Shared detection response construction."""

from __future__ import annotations

from app.ml.inference import ImageInferenceResult
from app.ml.metadata import ModelMetadata
from app.ml.video import VideoInferenceResult
from app.schemas.detection import DetectionResponse, VideoDetectionResponse


def image_response(
    result: ImageInferenceResult,
    metadata: ModelMetadata,
    elapsed_ms: int,
) -> DetectionResponse:
    return DetectionResponse(
        status="completed",
        prediction=result.prediction,
        verdict=result.verdict,  # type: ignore[arg-type]
        confidence_percent=result.confidence_percent,
        explanation=result.explanation,
        score=result.score,
        score_type=metadata.score_type,  # type: ignore[arg-type]
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        processing_time_ms=elapsed_ms,
        warnings=result.warnings,
        evidence={"face_detected": result.face_detected},
    )


def video_response(
    result: VideoInferenceResult,
    metadata: ModelMetadata,
    elapsed_ms: int,
) -> VideoDetectionResponse:
    return VideoDetectionResponse(
        status="completed",
        prediction=result.prediction,
        verdict=result.verdict,  # type: ignore[arg-type]
        confidence_percent=result.confidence_percent,
        explanation=result.explanation,
        score=result.score,
        score_type=metadata.score_type,  # type: ignore[arg-type]
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        processing_time_ms=elapsed_ms,
        warnings=result.warnings,
        evidence={"aggregation_notes": result.aggregation_method},
        frames_sampled=result.frames_sampled,
        frames_analyzed=result.frames_analyzed,
        frame_results=result.frame_results,
        aggregation_method=result.aggregation_method,
    )


def unavailable_image(reason: str, elapsed_ms: int = 0) -> DetectionResponse:
    return DetectionResponse(
        status="unavailable",
        prediction="analysis_unavailable",
        score=None,
        score_type="model_score",
        model_name="not_loaded",
        model_version="none",
        processing_time_ms=elapsed_ms,
        warnings=[reason],
    )


def unavailable_video(reason: str, elapsed_ms: int = 0) -> VideoDetectionResponse:
    return VideoDetectionResponse(
        status="unavailable",
        prediction="analysis_unavailable",
        score=None,
        score_type="model_score",
        model_name="not_loaded",
        model_version="none",
        processing_time_ms=elapsed_ms,
        warnings=[reason],
    )
