"""Request and response schemas for detection APIs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PredictionLabel = Literal[
    "likely_manipulated",
    "likely_authentic",
    "inconclusive",
    "analysis_unavailable",
]
VerdictLabel = Literal["REAL", "FAKE"]
JobStatus = Literal["queued", "processing", "completed", "failed"]
ScoreType = Literal["model_score", "calibrated_probability"]
AnalysisStatus = Literal["completed", "unavailable", "failed"]


class FrameResult(BaseModel):
    timestamp_sec: float
    prediction: PredictionLabel
    score: float | None = None
    face_detected: bool = False
    warning: str | None = None


class DetectionResponse(BaseModel):
    status: AnalysisStatus
    prediction: PredictionLabel
    verdict: VerdictLabel | None = None
    confidence_percent: float | None = None
    explanation: str | None = None
    score: float | None = None
    score_type: ScoreType = "model_score"
    model_name: str
    model_version: str
    processing_time_ms: int
    warnings: list[str] = Field(default_factory=list)
    evidence: dict | None = None


class VideoDetectionResponse(DetectionResponse):
    frames_sampled: int = 0
    frames_analyzed: int = 0
    frame_results: list[FrameResult] = Field(default_factory=list)
    aggregation_method: str = "median_manipulated_score_with_evidence_gate"


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    environment: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    checkpoint_present: bool
    reason: str | None = None


class ModelInfoResponse(BaseModel):
    loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    architecture: str | None = None
    device: str | None = None
    checkpoint_path: str | None = None
    score_type: ScoreType = "model_score"
    authentic_threshold: float | None = None
    manipulated_threshold: float | None = None
    classes: list[str] | None = None
    metrics: dict | None = None
    notes: list[str] = Field(default_factory=list)


class VideoJobCreated(BaseModel):
    job_id: str
    status: JobStatus


class VideoJobStatus(VideoDetectionResponse):
    job_id: str
    job_status: JobStatus
    progress: float = 0.0
    message: str | None = None
