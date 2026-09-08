from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PathoVerseTileDataset(Dataset):
    """
    PyTorch Dataset for extracted pathology tiles.

    By default, images are returned as float32 tensors in
    CHW format with values normalized to [0, 1].

    A custom transform can be supplied for model-specific
    preprocessing.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        transform: Optional[Callable] = None,
    ):
        self.manifest_path = Path(
            manifest_path
        )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: "
                f"{self.manifest_path}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        self.tiles = data.get(
            "tiles",
            [],
        )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int):

        record = self.tiles[index]

        image_path = Path(
            record["image_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Tile image not found: "
                f"{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        else:
            image = self._to_tensor(image)

        return {
            "image": image,
            "tile_id": record["tile_id"],
            "slide_id": record["slide_id"],
            "x": record["x"],
            "y": record["y"],
            "level": record["level"],
            "tissue_ratio": record[
                "tissue_ratio"
            ],
            "image_path": str(
                image_path
            ),
        }

    @staticmethod
    def _to_tensor(
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Convert PIL RGB image to float32 CHW tensor.

        Output range:
            [0, 1]
        """

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        tensor = torch.from_numpy(
            array
        )

        tensor = tensor.permute(
            2,
            0,
            1,
        )

        tensor = tensor / 255.0

        return tensor