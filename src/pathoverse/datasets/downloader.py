from __future__ import annotations

from pathlib import Path
from typing import Dict

from .medmnist import PathMNISTDataset
from .patchcamelyon import PatchCamelyonDataset


DATASET_ROOT = Path("data/raw")


def get_dataset(dataset_name: str):
    dataset_name = dataset_name.lower().strip()

    dataset_map = {
        "pathmnist": PathMNISTDataset(
            DATASET_ROOT / "medmnist"
        ),
        "patchcamelyon": PatchCamelyonDataset(
            DATASET_ROOT / "patchcamelyon"
        ),
    }

    if dataset_name not in dataset_map:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Available: {sorted(dataset_map)}"
        )

    return dataset_map[dataset_name]


def download_dataset(dataset_name: str) -> Dict:
    dataset = get_dataset(dataset_name)

    dataset.download()

    return {
        "dataset": dataset.name,
        "status": "downloaded",
        "root": str(dataset.root_dir),
    }