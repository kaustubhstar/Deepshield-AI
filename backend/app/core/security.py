"""Upload validation and safe temporary files."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": {".jpg", ".jpeg"},
    "image/png": {".png"},
    "image/webp": {".webp"},
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4": {".mp4"},
    "video/webm": {".webm"},
    "video/quicktime": {".mov"},
}


def _suffix(filename: str | None) -> str:
    return Path(filename or "").suffix.lower()


def validate_upload(
    file: UploadFile,
    *,
    allowed: dict[str, set[str]],
    max_bytes: int,
    kind: str,
) -> None:
    suffix = _suffix(file.filename)
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    allowed_suffixes = set().union(*allowed.values())
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported {kind} file extension. Allowed: {sorted(allowed_suffixes)}",
        )
    if content_type and content_type not in allowed and content_type != "application/octet-stream":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported {kind} content type: {content_type}",
        )


async def read_bounded(file: UploadFile, max_bytes: int, kind: str) -> bytes:
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{kind.capitalize()} exceeds the maximum size of {max_bytes} bytes.",
        )
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Empty {kind} upload.")
    return data


@contextmanager
def temp_media_file(suffix: str) -> Iterator[Path]:
    handle = tempfile.NamedTemporaryFile(prefix="deepshield_", suffix=suffix, delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
