from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .metadata import WSIMetadata
from .tiler import TileCoordinate


def build_manifest(
    metadata: WSIMetadata,
    tiles: Iterable[TileCoordinate],
    *,
    level: int,
    tile_size: int,
    stride: int,
    min_tissue_ratio: float,
) -> dict:

    tile_list = [
        tile.to_dict()
        for tile in tiles
    ]

    return {
        "schema_version": "1.0",
        "slide": {
            "slide_id": metadata.slide_id,
            "filename": metadata.filename,
            "format": metadata.format,
            "vendor": metadata.vendor,
            "dimensions": list(metadata.dimensions),
            "level_count": metadata.level_count,
            "mpp_x": metadata.mpp_x,
            "mpp_y": metadata.mpp_y,
            "objective_power": metadata.objective_power,
        },
        "tiling": {
            "coordinate_reference": "level_0",
            "level": level,
            "tile_size": tile_size,
            "stride": stride,
            "min_tissue_ratio": min_tissue_ratio,
        },
        "statistics": {
            "tile_count": len(tile_list),
        },
        "tiles": tile_list,
    }


def save_manifest(
    manifest: dict,
    path: str | Path,
) -> None:

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )