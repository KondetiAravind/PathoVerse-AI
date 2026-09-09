from pathlib import Path

from pathoverse.config import get_settings


settings = get_settings()


PROJECT_ROOT = settings.project_root

DATA_ROOT = settings.data_root
RAW_ROOT = settings.raw_root
PROCESSED_ROOT = settings.processed_root
METADATA_ROOT = settings.metadata_root
RESULTS_ROOT = settings.results_root
MODELS_ROOT = settings.models_root


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_root() -> Path:
    return DATA_ROOT


def get_processed_root() -> Path:
    return PROCESSED_ROOT


def get_metadata_root() -> Path:
    return METADATA_ROOT


def get_results_root() -> Path:
    return RESULTS_ROOT