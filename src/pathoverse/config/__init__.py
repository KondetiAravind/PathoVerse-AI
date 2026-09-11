from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ==========================================================
# SETTINGS
# ==========================================================

class Settings(BaseSettings):
    """
    Central application configuration.

    Environment variables use the PATHOVERSE_ prefix.

    Example:
        PATHOVERSE_PORT=9000
    """

    model_config = SettingsConfigDict(
        env_prefix="PATHOVERSE_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------
    # Application
    # ------------------------------------------------------

    app_name: str = "PathoVerse AI"

    app_version: str = "0.4.0"

    # PATHOVERSE_ENV is intentionally mapped to this field.
    environment: str = Field(
        default="development",
        validation_alias="PATHOVERSE_ENV",
    )

    # API key used in production API authentication.
    #
    # IMPORTANT:
    # Keep the real key in .env.production.
    # Never commit the real key to Git.
    api_key: str = ""

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

    cors_allow_methods: str = (
        "GET,POST,PUT,PATCH,DELETE,OPTIONS"
    )

    cors_allow_headers: str = "*"

    cors_origin_regex: str = r"https://.*\.vercel\.app"

    # ------------------------------------------------------
    # Project directories
    # ------------------------------------------------------

    project_root: Path = PROJECT_ROOT

    data_root: Path = (
        PROJECT_ROOT / "data"
    )

    raw_root: Path = (
        PROJECT_ROOT / "data" / "raw"
    )

    processed_root: Path = (
        PROJECT_ROOT / "data" / "processed"
    )

    metadata_root: Path = (
        PROJECT_ROOT / "data" / "metadata"
    )

    results_root: Path = (
        PROJECT_ROOT / "results"
    )

    models_root: Path = (
        PROJECT_ROOT / "models"
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    @field_validator("environment")
    @classmethod
    def validate_environment(
        cls,
        value: str,
    ) -> str:
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
    def validate_log_level(
        cls,
        value: str,
    ) -> str:
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
    # Parsed CORS helpers
    # ------------------------------------------------------

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def cors_method_list(self) -> list[str]:
        return [
            method.strip().upper()
            for method in self.cors_allow_methods.split(",")
            if method.strip()
        ]

    @property
    def cors_header_list(self) -> list[str]:
        return [
            header.strip()
            for header in self.cors_allow_headers.split(",")
            if header.strip()
        ]


# ==========================================================
# CACHED SETTINGS
# ==========================================================

@lru_cache
def get_settings() -> Settings:
    """
    Return the cached application settings.

    Caching ensures that every module uses the same
    configuration instance during the application lifecycle.
    """

    return Settings()