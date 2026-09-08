from __future__ import annotations

from typing import Dict

from .downloader import get_dataset


def validate_dataset(dataset_name: str) -> Dict:
    dataset = get_dataset(dataset_name)

    result = dataset.validate()

    return result