"""Image, video, and live-frame detection endpoints."""

from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile

from app.core.security import (
    ALLOWED_IMAGE_TYPES,
    ALLOWED_VIDEO_TYPES,
    read_bounded,
    temp_media_file,
    validate_upload,
)
from app.schemas.detection import DetectionResponse, VideoDetectionResponse, VideoJobCreated, VideoJobStatus

router = APIRouter(prefix="/api/detect")


@router.post("/image", response_model=DetectionResponse)
async def detect_image(request: Request, file: UploadFile = File(...)) -> DetectionResponse:
    settings = request.app.state.settings
    validate_upload(file, allowed=ALLOWED_IMAGE_TYPES, max_bytes=settings.max_image_bytes, kind="image")
    data = await read_bounded(file, settings.max_image_bytes, "image")
    return request.app.state.image_service.analyze_bytes(data, subject="image")


@router.post("/frame", response_model=DetectionResponse)
async def detect_frame(request: Request, file: UploadFile = File(...)) -> DetectionResponse:
    """Analyze a single captured camera frame. Frames are not stored."""
    settings = request.app.state.settings
    validate_upload(file, allowed=ALLOWED_IMAGE_TYPES, max_bytes=settings.max_image_bytes, kind="image")
    data = await read_bounded(file, settings.max_image_bytes, "image")
    return request.app.state.image_service.analyze_bytes(data, subject="camera frame")


@router.post("/video", response_model=VideoDetectionResponse)
async def detect_video(request: Request, file: UploadFile = File(...)) -> VideoDetectionResponse:
    settings = request.app.state.settings
    validate_upload(file, allowed=ALLOWED_VIDEO_TYPES, max_bytes=settings.max_video_bytes, kind="video")
    data = await read_bounded(file, settings.max_video_bytes, "video")
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with temp_media_file(suffix) as path:
        path.write_bytes(data)
        return request.app.state.video_service.analyze_path(path)


@router.post("/video/jobs", response_model=VideoJobCreated)
async def start_video_job(request: Request, file: UploadFile = File(...)) -> VideoJobCreated:
    settings = request.app.state.settings
    validate_upload(file, allowed=ALLOWED_VIDEO_TYPES, max_bytes=settings.max_video_bytes, kind="video")
    data = await read_bounded(file, settings.max_video_bytes, "video")
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    job_id = request.app.state.video_service.start_job(data, suffix)
    return VideoJobCreated(job_id=job_id, status="queued")


@router.get("/video/jobs/{job_id}", response_model=VideoJobStatus)
def video_job_status(request: Request, job_id: str) -> VideoJobStatus:
    job = request.app.state.video_service.get_job(job_id)
    return request.app.state.video_service.job_status_payload(job)
