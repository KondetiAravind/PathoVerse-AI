from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
METADATA_ROOT = DATA_ROOT / "metadata"
RESULTS_ROOT = PROJECT_ROOT / "results"
MODELS_ROOT = PROJECT_ROOT / "models"


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
