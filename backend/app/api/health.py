"""Health, readiness, and model metadata endpoints."""

from fastapi import APIRouter, Request

from app.schemas.detection import HealthResponse, ModelInfoResponse, ReadyResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(status="ok", service=settings.app_name, environment=settings.environment)


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    runtime = request.app.state.model_service.runtime
    return ReadyResponse(
        ready=runtime.loaded,
        model_loaded=runtime.loaded,
        checkpoint_present=runtime.checkpoint_present,
        reason=runtime.reason,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    settings = request.app.state.settings
    runtime = request.app.state.model_service.runtime
    metadata = runtime.metadata
    if not runtime.loaded or metadata is None:
        return ModelInfoResponse(
            loaded=False,
            checkpoint_path=str(settings.checkpoint_path),
            notes=[
                runtime.reason or "No compatible trained checkpoint is installed.",
                "Production inference stays unavailable until you train and install a checkpoint.",
            ],
        )
    return ModelInfoResponse(
        loaded=True,
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        architecture=metadata.architecture,
        device=str(runtime.device) if runtime.device else None,
        checkpoint_path=str(settings.checkpoint_path),
        score_type=metadata.score_type,  # type: ignore[arg-type]
        authentic_threshold=metadata.authentic_threshold,
        manipulated_threshold=metadata.manipulated_threshold,
        classes=metadata.classes,
        metrics=metadata.metrics,
        notes=metadata.notes,
    )
