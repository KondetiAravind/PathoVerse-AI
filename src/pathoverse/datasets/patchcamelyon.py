from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import BasePathologyDataset, DatasetInfo
from .registry import register_dataset


@register_dataset("patchcamelyon")
class PatchCamelyonDataset(BasePathologyDataset):
    """
    PatchCamelyon dataset adapter.

    The adapter is intentionally separated from the downloader so
    dataset acquisition and dataset logic remain independent.
    """

    def __init__(self, root_dir: str | Path):
        super().__init__(root_dir)

    @property
    def name(self) -> str:
        return "patchcamelyon"

    def download(self, **kwargs: Any) -> None:
        """
        Download PatchCamelyon using the official dataset distribution.

        The exact acquisition mechanism is kept separate so that the
        dataset can be replaced or mirrored without changing the
        PathoVerse dataset interface.
        """

        raise NotImplementedError(
            "PatchCamelyon acquisition will be enabled through the "
            "PathoVerse download manager."
        )

    def get_split(self, split: str):
        raise NotImplementedError(
            "PatchCamelyon split loading will be enabled after "
            "dataset acquisition."
        )

    def validate(self) -> Dict[str, Any]:
        return {
            "dataset": self.name,
            "valid": False,
            "status": "not_downloaded",
            "errors": [
                "PatchCamelyon data has not been acquired yet."
            ],
        }

    def statistics(self) -> Dict[str, Any]:
        return {
            "dataset": self.name,
            "status": "not_downloaded",
        }

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="patchcamelyon",
            description=(
                "Binary metastatic tissue classification dataset "
                "based on histopathology patches."
            ),
            num_classes=2,
            class_names=[
                "normal",
                "metastatic",
            ],
            image_shape=(96, 96, 3),
            split_names=[
                "train",
                "validation",
                "test",
            ],
            root_dir=self.root_dir,
        )