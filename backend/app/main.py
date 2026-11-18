"""DeepShield AI FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.detection import router as detection_router
from app.api.health import router as health_router
from app.config import get_settings
from app.core.logging import configure_logging
from app.services.image_service import ImageService
from app.services.model_service import ModelService
from app.services.video_service import VideoService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    model_service = ModelService(settings)
    model_service.load()
    app.state.settings = settings
    app.state.model_service = model_service
    app.state.image_service = ImageService(settings, model_service)
    app.state.video_service = VideoService(settings, model_service)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_per_minute])
    application = FastAPI(
        title="DeepShield AI API",
        version="0.1.0",
        description="Real inference API for DeepShield AI. Returns analysis_unavailable until a trained checkpoint is installed.",
        lifespan=lifespan,
    )
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    if settings.environment.lower() != "test":
        application.add_middleware(SlowAPIMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    application.include_router(health_router)
    application.include_router(detection_router)

    @application.exception_handler(StarletteHTTPException)
    async def http_handler(_request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @application.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": "Invalid request format."})

    @application.exception_handler(Exception)
    async def unhandled_handler(_request: Request, _exc: Exception):
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    return application


app = create_app()
