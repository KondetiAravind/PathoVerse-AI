from pathlib import Path

from pathoverse.config import get_settings


settings = get_settings()

PROJECT_ROOT: Path = settings.project_root
DATA_ROOT: Path = settings.data_root
RAW_ROOT: Path = settings.raw_root
PROCESSED_ROOT: Path = settings.processed_root
METADATA_ROOT: Path = settings.metadata_root
RESULTS_ROOT: Path = settings.results_root
MODELS_ROOT: Path = settings.models_root


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_root() -> Path:
    return DATA_ROOT


def get_raw_root() -> Path:
    return RAW_ROOT


def get_processed_root() -> Path:
    return PROCESSED_ROOT


def get_metadata_root() -> Path:
    return METADATA_ROOT


def get_results_root() -> Path:
    return RESULTS_ROOT


def get_models_root() -> Path:
    return MODELS_ROOT