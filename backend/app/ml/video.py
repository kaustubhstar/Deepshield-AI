"""Video frame sampling, per-frame inference, and documented aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.ml.inference import ImageDetector
from app.ml.metadata import ModelMetadata
from app.ml.preprocess import MediaError
from app.ml.scores import decision_from_manipulated_score
from app.schemas.detection import FrameResult, PredictionLabel

AGGREGATION_METHOD = "median_manipulated_score_with_evidence_gate"
AGGREGATION_NOTES = (
    "The video label uses the middle FAKE score from the sampled frames. "
    "Nearby frames look alike, so this is a model confidence score, not proof."
)


@dataclass
class VideoInferenceResult:
    prediction: PredictionLabel
    score: float | None
    frames_sampled: int
    frames_analyzed: int
    frame_results: list[FrameResult]
    warnings: list[str] = field(default_factory=list)
    aggregation_method: str = AGGREGATION_METHOD
    verdict: str | None = None
    confidence_percent: float | None = None
    explanation: str | None = None


def _open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MediaError("The video could not be decoded. Use MP4 or WebM if possible.")
    return capture


def video_duration_seconds(path: Path) -> float:
    capture = _open_capture(path)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps > 0:
            return float(frame_count / fps)
        return 0.0
    finally:
        capture.release()


def sample_frame_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    count = min(max_frames, frame_count)
    if count == 1:
        return [0]
    return [int(round(i * (frame_count - 1) / (count - 1))) for i in range(count)]


def analyze_video(
    path: Path,
    detector: ImageDetector,
    metadata: ModelMetadata,
    max_frames: int,
    max_seconds: float,
    min_frames_for_decision: int,
    progress_callback=None,
) -> VideoInferenceResult:
    capture = _open_capture(path)
    warnings: list[str] = [AGGREGATION_NOTES]
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps else 0.0
        if duration > max_seconds:
            raise MediaError(f"Video is longer than the configured limit of {max_seconds:.0f} seconds.")
        if metadata.authentic_index == metadata.manipulated_index:
            raise MediaError("Model class indices are invalid, so no label was created.")
        indices = sample_frame_indices(frame_count, max_frames)
        if not indices:
            raise MediaError("No frames could be sampled from this video.")

        frame_results: list[FrameResult] = []
        scores: list[float] = []
        for step, index in enumerate(indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            timestamp = index / fps if fps else float(index)
            if not ok or bgr is None:
                frame_results.append(
                    FrameResult(
                        timestamp_sec=round(timestamp, 3),
                        prediction="inconclusive",
                        score=None,
                        face_detected=False,
                        warning="Frame could not be decoded.",
                    )
                )
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                result = detector.analyze(image, subject="camera frame")
                frame_results.append(
                    FrameResult(
                        timestamp_sec=round(timestamp, 3),
                        prediction=result.prediction,
                        score=result.score,
                        face_detected=result.face_detected,
                        warning=None if result.face_detected else "No face detected on this frame.",
                    )
                )
                scores.append(result.score)
            if progress_callback:
                progress_callback(step / len(indices))

        analyzed = len(scores)
        if analyzed == 0:
            return VideoInferenceResult(
                prediction="inconclusive",
                score=None,
                frames_sampled=len(indices),
                frames_analyzed=0,
                frame_results=frame_results,
                warnings=warnings + ["No usable frames were analyzed."],
                explanation="No readable frames were found, so no REAL or FAKE result was created.",
            )
        median_score = float(np.median(np.array(scores, dtype=np.float64)))
        if analyzed < min_frames_for_decision:
            warnings.append(
                f"Only {analyzed} frames could be checked. At least {min_frames_for_decision} are needed before a video result is shown."
            )
            return VideoInferenceResult(
                prediction="inconclusive",
                score=median_score,
                frames_sampled=len(indices),
                frames_analyzed=analyzed,
                frame_results=frame_results,
                warnings=warnings,
                explanation="Not enough readable frames to choose REAL or FAKE. No result was invented.",
            )
        decision = decision_from_manipulated_score(median_score, "video")
        return VideoInferenceResult(
            prediction=decision.prediction,
            score=median_score,
            frames_sampled=len(indices),
            frames_analyzed=analyzed,
            frame_results=frame_results,
            warnings=warnings,
            verdict=decision.verdict,
            confidence_percent=decision.confidence_percent,
            explanation=decision.explanation,
        )
    finally:
        capture.release()
