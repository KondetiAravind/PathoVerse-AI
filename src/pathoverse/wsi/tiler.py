from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .reader import WSIReader
from .tissue import resize_mask


@dataclass(frozen=True)
class TileCoordinate:
    tile_id: int
    x: int
    y: int
    width: int
    height: int
    tissue_ratio: float

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "tissue_ratio": round(
                self.tissue_ratio,
                6,
            ),
        }


class TileCoordinateEngine:
    """
    Generates level-0 WSI tile coordinates using a tissue mask.
    """

    def __init__(
        self,
        *,
        tile_size: int = 224,
        stride: int | None = None,
        min_tissue_ratio: float = 0.5,
    ):
        if tile_size <= 0:
            raise ValueError(
                "tile_size must be positive."
            )

        if stride is None:
            stride = tile_size

        if stride <= 0:
            raise ValueError(
                "stride must be positive."
            )

        if not 0.0 <= min_tissue_ratio <= 1.0:
            raise ValueError(
                "min_tissue_ratio must be between 0 and 1."
            )

        self.tile_size = tile_size
        self.stride = stride
        self.min_tissue_ratio = min_tissue_ratio

    def generate(
        self,
        reader: WSIReader,
        tissue_mask: np.ndarray,
    ) -> list[TileCoordinate]:

        level0_width, level0_height = reader.dimensions

        mask = resize_mask(
            tissue_mask,
            level0_width,
            level0_height,
        )

        tiles: list[TileCoordinate] = []

        tile_id = 0

        for y in range(
            0,
            level0_height - self.tile_size + 1,
            self.stride,
        ):
            for x in range(
                0,
                level0_width - self.tile_size + 1,
                self.stride,
            ):

                tile_mask = mask[
                    y : y + self.tile_size,
                    x : x + self.tile_size,
                ]

                tissue_ratio = float(
                    tile_mask.mean()
                )

                if (
                    tissue_ratio
                    < self.min_tissue_ratio
                ):
                    continue

                tiles.append(
                    TileCoordinate(
                        tile_id=tile_id,
                        x=x,
                        y=y,
                        width=self.tile_size,
                        height=self.tile_size,
                        tissue_ratio=tissue_ratio,
                    )
                )

                tile_id += 1

        return tiles