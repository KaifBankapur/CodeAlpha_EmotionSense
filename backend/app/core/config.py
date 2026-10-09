"""Application settings, loaded from environment variables.

Everything configurable without touching code lives here. The defaults are
deliberately safe for local development:

* CORS is **not** open. ``cors_origins`` defaults to the two Vite dev ports and
  must be set explicitly for any other origin.
* No secrets are needed to run the service, so there is nothing to leak; the
  settings class simply has no fields that would encourage adding some.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.config import AudioConfig


class Settings(BaseSettings):
    """Environment-driven configuration (``SER_*`` prefix)."""

    model_config = SettingsConfigDict(
        env_prefix="SER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Speech Emotion Recognition API"
    version: str = "1.0.0"
    description: str = (
        "Detects the emotional state of a speaker from audio, using MFCC "
        "features and a CNN + BiLSTM classifier trained on RAVDESS."
    )

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    #: Explicit run name/directory to serve. Empty -> follow
    #: ``artifacts/active_model.json``.
    model_run: str | None = None

    #: Upload ceiling in bytes. Defaults to the AudioConfig value (20 MB) so the
    #: API and the ML pipeline agree on what a valid upload is.
    max_upload_bytes: int = Field(
        default=AudioConfig().max_upload_bytes, ge=1024, le=200 * 1024 * 1024
    )

    #: Comma-separated list of allowed browser origins.
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    #: Fail startup when no trained model is available. Keeping this ``True``
    #: means a misconfigured deployment fails loudly at boot instead of
    #: returning 500 on every prediction.
    require_model_on_startup: bool = True

    #: Concurrent inference requests. Each forward pass is CPU-bound, so more
    #: concurrency than this just makes every request slower.
    max_concurrent_predictions: int = Field(default=4, ge=1, le=64)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept ``SER_CORS_ORIGINS=a,b`` as well as a JSON list."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("cors_origins")
    @classmethod
    def _reject_wildcard(cls, value: list[str]) -> list[str]:
        """Refuse ``*``.

        This API sends no cookies and no ``Authorization`` header, so a wildcard
        would "work" - which is exactly the problem. It reads as permission
        granted to every site on the internet, and the moment someone adds
        credentials the browser rejects the response anyway. Failing at startup
        turns a latent misconfiguration into an explicit one.
        """
        if any(origin.strip() == "*" for origin in value):
            raise ValueError(
                "cors_origins must list explicit origins, not '*'. This API "
                "does not use cookies or credentials, so a wildcard adds "
                "cross-origin access without adding any capability."
            )
        return value

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
            raise ValueError(f"Unsupported log level: {value}")
        return level

    def audio_config(self) -> AudioConfig:
        """Audio settings for the API, with the upload limit applied."""
        cfg = AudioConfig()
        cfg.max_upload_bytes = self.max_upload_bytes
        return cfg


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor so the environment is parsed once per process."""
    return Settings()


__all__ = ["Settings", "get_settings"]
