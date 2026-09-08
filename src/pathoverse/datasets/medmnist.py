from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .base import BasePathologyDataset, DatasetInfo
from .registry import register_dataset


@register_dataset("pathmnist")
class PathMNISTDataset(BasePathologyDataset):
    """
    PathMNIST adapter for PathoVerse AI.

    PathMNIST is a 9-class histopathology classification dataset
    derived from colorectal cancer tissue images.
    """

    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir)

        self._dataset = None

    @property
    def name(self) -> str:
        return "pathmnist"

    def download(self, **kwargs: Any) -> None:
        import medmnist
        from medmnist import INFO

        data_flag = "pathmnist"

        info = INFO[data_flag]
        DataClass = getattr(medmnist, info["python_class"])

        self._dataset = DataClass(
            split="train",
            root=str(self.root_dir),
            download=True,
        )

        # Download validation and test splits as well.
        DataClass(
            split="val",
            root=str(self.root_dir),
            download=True,
        )

        DataClass(
            split="test",
            root=str(self.root_dir),
            download=True,
        )

    def _load_split(self, split: str):
        import medmnist
        from medmnist import INFO

        data_flag = "pathmnist"

        info = INFO[data_flag]
        DataClass = getattr(medmnist, info["python_class"])

        return DataClass(
            split=split,
            root=str(self.root_dir),
            download=False,
        )

    def get_split(self, split: str):
        valid_splits = {"train", "val", "test"}

        if split not in valid_splits:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of {sorted(valid_splits)}."
            )

        return self._load_split(split)

    def validate(self) -> Dict[str, Any]:
        results = {
            "dataset": self.name,
            "valid": True,
            "splits": {},
            "errors": [],
        }

        for split in ("train", "val", "test"):
            try:
                dataset = self.get_split(split)

                length = len(dataset)

                if length == 0:
                    results["valid"] = False
                    results["errors"].append(
                        f"{split} split is empty."
                    )

                results["splits"][split] = {
                    "num_samples": length,
                }

            except Exception as exc:
                results["valid"] = False
                results["errors"].append(
                    f"{split}: {exc}"
                )

        return results

    def statistics(self) -> Dict[str, Any]:
        stats = {
            "dataset": self.name,
            "splits": {},
        }

        for split in ("train", "val", "test"):
            dataset = self.get_split(split)

            labels = np.asarray(dataset.labels).reshape(-1)

            unique, counts = np.unique(
                labels,
                return_counts=True,
            )

            stats["splits"][split] = {
                "num_samples": len(dataset),
                "num_classes_present": len(unique),
                "class_distribution": {
                    str(int(label)): int(count)
                    for label, count in zip(unique, counts)
                },
            }

        return stats

    def info(self) -> DatasetInfo:
        import medmnist
        from medmnist import INFO

        info = INFO["pathmnist"]

        class_names = list(info["label"].values())

        return DatasetInfo(
            name="pathmnist",
            description=(
                "9-class colorectal histopathology image "
                "classification dataset."
            ),
            num_classes=9,
            class_names=class_names,
            image_shape=(28, 28, 3),
            split_names=["train", "val", "test"],
            root_dir=self.root_dir,
        )