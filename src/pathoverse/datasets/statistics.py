from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .downloader import get_dataset


def compute_dataset_statistics(dataset_name: str) -> Dict:
    dataset = get_dataset(dataset_name)

    return dataset.statistics()


def save_statistics(
    dataset_name: str,
    output_path: str | Path,
) -> Path:

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics = compute_dataset_statistics(dataset_name)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            statistics,
            file,
            indent=2,
        )

    return output_path