from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ==========================================================
# APPLICATION SETTINGS
# ==========================================================


class Settings(BaseSettings):
    """
    Centralized runtime configuration for PathoVerse AI.

    Configuration can be supplied through environment variables
    or a local .env file.

    Environment variables use the PATHOVERSE_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="PATHOVERSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------
    # Application
    # ------------------------------------------------------

    app_name: str = "PathoVerse AI"

    app_version: str = "0.4.0"

    environment: str = "development"

    # ------------------------------------------------------
    # Server
    # ------------------------------------------------------

    host: str = "0.0.0.0"

    port: int = 8000

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------

    log_level: str = "INFO"

    # ------------------------------------------------------
    # CORS
    # ------------------------------------------------------

    cors_origins: str = (
        "http://localhost:3000,"
        "http://127.0.0.1:3000"
    )

    cors_allow_credentials: bool = True

    # ------------------------------------------------------
    # Project Paths
    # ------------------------------------------------------

    project_root: Path = PROJECT_ROOT

    data_root: Path = PROJECT_ROOT / "data"

    raw_root: Path = PROJECT_ROOT / "data" / "raw"

    processed_root: Path = PROJECT_ROOT / "data" / "processed"

    metadata_root: Path = PROJECT_ROOT / "data" / "metadata"

    results_root: Path = PROJECT_ROOT / "results"

    models_root: Path = PROJECT_ROOT / "models"

    # ------------------------------------------------------
    # Validators
    # ------------------------------------------------------

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        value = value.strip().lower()

        allowed = {
            "development",
            "staging",
            "production",
            "test",
        }

        if value not in allowed:
            raise ValueError(
                f"Invalid environment '{value}'. "
                f"Expected one of: {sorted(allowed)}"
            )

        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.strip().upper()

        allowed = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if value not in allowed:
            raise ValueError(
                f"Invalid log level '{value}'. "
                f"Expected one of: {sorted(allowed)}"
            )

        return value

    # ------------------------------------------------------
    # Normalized CORS Origins
    # ------------------------------------------------------

    @property
    def cors_origin_list(self) -> list[str]:
        """
        Return configured CORS origins as a normalized list.
        """

        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


# ==========================================================
# SETTINGS ACCESSOR
# ==========================================================


@lru_cache
def get_settings() -> Settings:
    """
    Return the cached application settings instance.

    Caching ensures that the configuration is created only
    once during the lifetime of the process.
    """

    return Settings()