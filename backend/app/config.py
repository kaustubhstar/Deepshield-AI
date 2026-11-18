"""Environment-driven application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "DeepShield AI"
    environment: str = "development"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    model_dir: Path = BACKEND_ROOT / "models"
    checkpoint_filename: str = "deepshield_efficientnet_b0.pt"
    metadata_filename: str = "model_metadata.json"

    device: str = "cpu"
    inference_max_concurrency: int = 1
    request_timeout_seconds: float = 120.0

    max_image_bytes: int = 8 * 1024 * 1024
    max_video_bytes: int = 80 * 1024 * 1024
    max_image_pixels: int = 4096 * 4096
    min_image_side: int = 32
    max_video_seconds: float = 120.0
    video_max_frames: int = 24
    video_job_ttl_seconds: int = 1800

    authentic_threshold: float = 0.35
    manipulated_threshold: float = 0.65
    min_video_frames_for_decision: int = 3

    rate_limit_per_minute: str = "30/minute"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / self.checkpoint_filename

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / self.metadata_filename


@lru_cache
def get_settings() -> Settings:
    return Settings()
