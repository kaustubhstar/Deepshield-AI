"""Synchronous short-video analysis and in-memory background jobs."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import HTTPException, status

from app.config import Settings
from app.ml.preprocess import MediaError
from app.ml.video import analyze_video, video_duration_seconds
from app.schemas.detection import JobStatus, VideoDetectionResponse, VideoJobStatus
from app.services.model_service import ModelService
from app.services.score_service import unavailable_video, video_response


@dataclass
class VideoJob:
    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: VideoDetectionResponse | None = None


class VideoService:
    def __init__(self, settings: Settings, model_service: ModelService) -> None:
        self.settings = settings
        self.model_service = model_service
        self._jobs: dict[str, VideoJob] = {}
        self._guard = threading.Lock()

    def _purge(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.settings.video_job_ttl_seconds)
        expired = [job_id for job_id, job in self._jobs.items() if job.created_at < cutoff]
        for job_id in expired:
            self._jobs.pop(job_id, None)

    def analyze_path(self, path: Path, progress_callback=None) -> VideoDetectionResponse:
        start = time.perf_counter()
        if not self.model_service.runtime.loaded:
            return unavailable_video(self.model_service.runtime.reason or "Checkpoint not installed.")
        try:
            duration = video_duration_seconds(path)
            if duration > self.settings.max_video_seconds:
                raise MediaError(
                    f"Video duration {duration:.1f}s exceeds the limit of {self.settings.max_video_seconds:.0f}s."
                )
            detector = self.model_service.require_detector()
            assert self.model_service.runtime.metadata is not None
            assert self.model_service._lock is not None
            with self.model_service._lock:
                result = analyze_video(
                    path,
                    detector,
                    self.model_service.runtime.metadata,
                    max_frames=self.settings.video_max_frames,
                    max_seconds=self.settings.max_video_seconds,
                    min_frames_for_decision=self.settings.min_video_frames_for_decision,
                    progress_callback=progress_callback,
                )
        except MediaError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return video_response(result, self.model_service.runtime.metadata, elapsed_ms)

    def start_job(self, data: bytes, suffix: str) -> str:
        job_id = str(uuid.uuid4())
        job = VideoJob(job_id=job_id, status="queued", message="Queued")
        with self._guard:
            self._purge()
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job_id, data, suffix), daemon=True)
        thread.start()
        return job_id

    def get_job(self, job_id: str) -> VideoJob:
        with self._guard:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job_id")
        return job

    def job_status_payload(self, job: VideoJob) -> VideoJobStatus:
        base = job.result or unavailable_video(job.message or "Job is still running.")
        if job.result is None:
            base = VideoDetectionResponse(
                status="unavailable" if job.status == "failed" else "unavailable",
                prediction="analysis_unavailable" if job.status != "completed" else "inconclusive",
                score=None,
                score_type="model_score",
                model_name=self.model_service.runtime.metadata.model_name
                if self.model_service.runtime.metadata
                else "not_loaded",
                model_version=self.model_service.runtime.metadata.model_version
                if self.model_service.runtime.metadata
                else "none",
                processing_time_ms=0,
                warnings=[job.message] if job.message else [],
            )
            if job.status == "failed":
                base.status = "failed"
        return VideoJobStatus(
            **base.model_dump(),
            job_id=job.job_id,
            job_status=job.status,
            progress=job.progress,
            message=job.message,
        )

    def _run_job(self, job_id: str, data: bytes, suffix: str) -> None:
        from app.core.security import temp_media_file

        with self._guard:
            job = self._jobs[job_id]
            job.status = "processing"
            job.message = "Analyzing frames"
        try:
            with temp_media_file(suffix) as path:
                path.write_bytes(data)

                def on_progress(value: float) -> None:
                    with self._guard:
                        self._jobs[job_id].progress = value

                result = self.analyze_path(path, progress_callback=on_progress)
            with self._guard:
                job = self._jobs[job_id]
                job.status = "completed"
                job.progress = 1.0
                job.result = result
                job.message = "Completed"
        except HTTPException as exc:
            with self._guard:
                job = self._jobs[job_id]
                job.status = "failed"
                job.message = str(exc.detail)
                job.result = unavailable_video(str(exc.detail))
                job.result.status = "failed"
        except Exception:  # noqa: BLE001
            with self._guard:
                job = self._jobs[job_id]
                job.status = "failed"
                job.message = "Video analysis failed."
                job.result = unavailable_video("Video analysis failed.")
                job.result.status = "failed"
