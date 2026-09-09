from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import openslide
from PIL import Image

from pathoverse.api.dependencies import (
    get_data_root,
    get_metadata_root,
    get_processed_root,
)


class WSIService:
    """Read-only service for PathoVerse WSI artifacts."""

    def __init__(self) -> None:
        self.data_root = get_data_root()
        self.metadata_root = get_metadata_root()
        self.processed_root = get_processed_root()

        self.wsi_root = self.processed_root / "wsi"

        self.tile_manifest = (
            self.metadata_root
            / "tiles"
            / "extracted_tiles.json"
        )

    def _metadata_path(
        self,
        slide_id: str,
    ) -> Path:
        return (
            self.wsi_root
            / f"{slide_id}_metadata.json"
        )

    def _thumbnail_path(
        self,
        slide_id: str,
    ) -> Path:
        return (
            self.wsi_root
            / f"{slide_id}_thumbnail.jpg"
        )

    def _mask_path(
        self,
        slide_id: str,
    ) -> Path:
        return (
            self.wsi_root
            / f"{slide_id}_tissue_mask.png"
        )

    def _tiles_path(
        self,
        slide_id: str,
    ) -> Path:
        return (
            self.wsi_root
            / f"{slide_id}_tiles.json"
        )

    def _calculate_tissue_ratio(
        self,
        slide_id: str,
    ) -> float | None:
        path = self._mask_path(slide_id)

        if not path.exists():
            return None

        try:
            with Image.open(path) as image:
                mask = np.asarray(
                    image.convert("L"),
                    dtype=np.uint8,
                )

                if mask.size == 0:
                    return None

                tissue_pixels = np.count_nonzero(
                    mask > 0
                )

                return float(
                    tissue_pixels / mask.size
                )

        except Exception:
            return None

    def _get_tile_count(
        self,
        slide_id: str,
    ) -> int:
        try:
            return len(self.get_tiles(slide_id))
        except FileNotFoundError:
            return 0

    def _enrich_slide_metrics(
        self,
        data: dict[str, Any],
        slide_id: str,
    ) -> dict[str, Any]:
        enriched = dict(data)

        if enriched.get("tissue_ratio") is None:
            enriched["tissue_ratio"] = (
                self._calculate_tissue_ratio(
                    slide_id
                )
            )

        if enriched.get("tile_count") is None:
            enriched["tile_count"] = (
                self._get_tile_count(slide_id)
            )

        return enriched

    def list_slides(
        self,
    ) -> list[dict[str, Any]]:
        slides = []

        for path in sorted(
            self.wsi_root.glob(
                "*_metadata.json"
            )
        ):
            data = self._load_json(path)

            slide_id = data.get(
                "slide_id",
                path.stem.replace(
                    "_metadata",
                    "",
                ),
            )

            data = self._enrich_slide_metrics(
                data,
                slide_id,
            )

            dimensions = data.get(
                "dimensions",
                [
                    data.get("width", 0),
                    data.get("height", 0),
                ],
            )

            slides.append(
                {
                    "slide_id": slide_id,
                    "filename": data.get(
                        "filename",
                        path.name,
                    ),
                    "format": data.get("format"),
                    "vendor": data.get("vendor"),
                    "width": dimensions[0],
                    "height": dimensions[1],
                    "levels": data.get(
                        "level_count",
                        data.get("levels", 0),
                    ),
                    "mpp_x": data.get("mpp_x"),
                    "mpp_y": data.get("mpp_y"),
                    "objective_power": data.get(
                        "objective_power"
                    ),
                }
            )

        return slides

    def get_slide(
        self,
        slide_id: str,
    ) -> dict[str, Any]:
        path = self._metadata_path(slide_id)

        if not path.exists():
            raise FileNotFoundError(
                f"Slide '{slide_id}' not found."
            )

        data = self._load_json(path)

        return self._enrich_slide_metrics(
            data,
            slide_id,
        )

    def get_thumbnail_path(
        self,
        slide_id: str,
    ) -> Path:
        path = self._thumbnail_path(slide_id)

        if not path.exists():
            raise FileNotFoundError(
                f"Thumbnail for '{slide_id}' not found."
            )

        return path

    def get_mask_path(
        self,
        slide_id: str,
    ) -> Path:
        path = self._mask_path(slide_id)

        if not path.exists():
            raise FileNotFoundError(
                f"Tissue mask for '{slide_id}' not found."
            )

        return path

    def get_tiles(
        self,
        slide_id: str,
    ) -> list[dict[str, Any]]:
        path = self._tiles_path(slide_id)

        if path.exists():
            data = self._load_json(path)

            if isinstance(data, dict):
                return data.get(
                    "tiles",
                    [],
                )

            if isinstance(data, list):
                return data

        if self.tile_manifest.exists():
            data = self._load_json(
                self.tile_manifest
            )

            return [
                tile
                for tile in data.get(
                    "tiles",
                    [],
                )
                if tile.get("slide_id")
                == slide_id
            ]

        raise FileNotFoundError(
            f"Tile manifest for '{slide_id}' not found."
        )

    def get_tile(
        self,
        slide_id: str,
        tile_id: int,
    ) -> dict[str, Any]:
        tiles = self.get_tiles(slide_id)

        for tile in tiles:
            if int(tile["tile_id"]) == int(tile_id):
                return tile

        raise FileNotFoundError(
            f"Tile {tile_id} for slide "
            f"'{slide_id}' not found."
        )

    def get_tile_image(
        self,
        slide_id: str,
        tile_id: int,
    ) -> bytes:
        tile = self.get_tile(
            slide_id,
            tile_id,
        )

        x = int(tile["x"])
        y = int(tile["y"])
        width = int(tile["width"])
        height = int(tile["height"])

        slide_path = (
            self.data_root
            / "raw"
            / "wsi"
            / f"{slide_id}.svs"
        )

        if not slide_path.exists():
            raise FileNotFoundError(
                f"WSI file not found: {slide_path}"
            )

        try:
            slide = openslide.OpenSlide(
                str(slide_path)
            )

            try:
                image = slide.read_region(
                    (x, y),
                    0,
                    (width, height),
                ).convert("RGB")
            finally:
                slide.close()

        except Exception as exc:
            raise RuntimeError(
                f"Failed to read tile {tile_id} "
                f"from slide '{slide_id}': {exc}"
            ) from exc

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG",
        )

        return buffer.getvalue()


    def get_thumbnail_size(
        self,
        slide_id: str,
    ) -> tuple[int, int]:
        path = self.get_thumbnail_path(
            slide_id
        )

        with Image.open(path) as image:
            return image.size

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)


wsi_service = WSIService()