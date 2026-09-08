from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pathoverse.wsi import TileCoordinate, WSIReader

from .quality import TileQualityChecker


@dataclass(frozen=True)
class ExtractedTile:
    tile_id: int
    slide_id: str
    x: int
    y: int
    width: int
    height: int
    level: int
    tissue_ratio: float
    image_path: str
    qc: dict

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "slide_id": self.slide_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "level": self.level,
            "tissue_ratio": round(
                self.tissue_ratio,
                6,
            ),
            "image_path": self.image_path,
            "qc": self.qc,
        }


class TileExtractor:
    """
    Extracts WSI tiles from a tile coordinate manifest.

    Coordinates are always interpreted in the level-0 reference frame.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        level: int = 0,
        quality_checker: TileQualityChecker | None = None,
        image_format: str = "png",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.level = level
        self.quality_checker = (
            quality_checker
            if quality_checker is not None
            else TileQualityChecker()
        )

        image_format = image_format.lower()

        if image_format not in {"png", "jpg", "jpeg"}:
            raise ValueError(
                "image_format must be png, jpg, or jpeg."
            )

        self.image_format = image_format

    def extract(
        self,
        reader: WSIReader,
        tiles: Iterable[dict | TileCoordinate],
    ) -> tuple[list[ExtractedTile], list[dict]]:

        extracted: list[ExtractedTile] = []
        rejected: list[dict] = []

        for tile in tiles:

            if isinstance(tile, TileCoordinate):
                tile_id = tile.tile_id
                x = tile.x
                y = tile.y
                width = tile.width
                height = tile.height
                tissue_ratio = tile.tissue_ratio
            else:
                tile_id = int(tile["tile_id"])
                x = int(tile["x"])
                y = int(tile["y"])
                width = int(tile["width"])
                height = int(tile["height"])
                tissue_ratio = float(
                    tile["tissue_ratio"]
                )

            image = reader.read_region(
                x,
                y,
                width,
                height,
                level=self.level,
                convert="RGB",
            )

            qc_result = self.quality_checker.check(
                image,
                tissue_ratio,
            )

            base_record = {
                "tile_id": tile_id,
                "slide_id": reader.metadata.slide_id,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "level": self.level,
                "tissue_ratio": tissue_ratio,
                "qc": qc_result.to_dict(),
            }

            if not qc_result.passed:
                rejected.append(
                    {
                        **base_record,
                        "reason": qc_result.reason,
                    }
                )
                continue

            filename = (
                f"{reader.metadata.slide_id}"
                f"_L{self.level}"
                f"_X{x:06d}"
                f"_Y{y:06d}"
                f".{self.image_format}"
            )

            output_path = (
                self.output_dir / filename
            )

            save_kwargs = {}

            if self.image_format in {
                "jpg",
                "jpeg",
            }:
                save_kwargs["quality"] = 95

            image.save(
                output_path,
                **save_kwargs,
            )

            extracted.append(
                ExtractedTile(
                    tile_id=tile_id,
                    slide_id=reader.metadata.slide_id,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    level=self.level,
                    tissue_ratio=tissue_ratio,
                    image_path=str(
                        output_path
                    ),
                    qc=qc_result.to_dict(),
                )
            )

        return extracted, rejected


def save_tile_manifest(
    tiles: list[ExtractedTile],
    rejected: list[dict],
    path: str | Path,
) -> None:

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "schema_version": "1.0",
        "statistics": {
            "extracted_count": len(tiles),
            "rejected_count": len(rejected),
            "total_count": len(tiles) + len(rejected),
        },
        "tiles": [
            tile.to_dict()
            for tile in tiles
        ],
        "rejected_tiles": rejected,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )