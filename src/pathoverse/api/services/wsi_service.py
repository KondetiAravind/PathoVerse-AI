from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pathoverse.api.dependencies import get_processed_root


logger = logging.getLogger("pathoverse.api.wsi")


class WSIService:
    """
    Service layer for Whole-Slide Image metadata,
    tile manifests, thumbnails, tissue masks and
    extracted tile images.
    """

    def __init__(self) -> None:
        self.processed_root = get_processed_root()

        self.wsi_root = self.processed_root / "wsi"
        self.tile_root = self.processed_root / "tiles"

    # ======================================================
    # SLIDES
    # ======================================================

    def list_slides(self) -> list[dict[str, Any]]:
        """
        Return all available WSI slides.
        """

        if not self.wsi_root.exists():
            return []

        metadata_files = sorted(
            self.wsi_root.glob("*_metadata.json")
        )

        slides: list[dict[str, Any]] = []

        for metadata_path in metadata_files:
            try:
                metadata = self._read_json(metadata_path)

                slide = self._build_slide_metadata(metadata)

                slide_id = slide["slide_id"]

                manifest = self._load_tile_manifest(
                    slide_id,
                    required=False,
                )

                if manifest is not None:
                    tiles = manifest.get(
                        "tiles",
                        [],
                    )

                    slide["tile_count"] = len(tiles)

                tissue_ratio = self._get_slide_tissue_ratio(
                    slide_id
                )

                if tissue_ratio is not None:
                    slide["tissue_ratio"] = tissue_ratio

                slides.append(slide)

            except Exception:
                logger.exception(
                    "Failed to parse WSI metadata: %s",
                    metadata_path,
                )

        return slides

    # ======================================================
    # SLIDE DETAIL
    # ======================================================

    def get_slide(
        self,
        slide_id: str,
    ) -> dict[str, Any]:
        """
        Return detailed WSI metadata.
        """

        metadata_path = (
            self.wsi_root
            / f"{slide_id}_metadata.json"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Slide metadata not found: {slide_id}"
            )

        metadata = self._read_json(
            metadata_path
        )

        slide = self._build_slide_metadata(
            metadata
        )

        manifest = self._load_tile_manifest(
            slide_id,
            required=True,
        )

        tiles = manifest.get(
            "tiles",
            [],
        )

        slide["tile_count"] = len(tiles)

        slide["tiling"] = manifest.get(
            "tiling",
            {},
        )

        slide["statistics"] = manifest.get(
            "statistics",
            {},
        )

        # --------------------------------------------------
        # WHOLE-SLIDE TISSUE RATIO
        #
        # This is calculated from the generated tissue mask.
        #
        # We intentionally DO NOT average tile tissue ratios
        # because the tile manifest contains only tissue-rich
        # tiles that passed the minimum tissue threshold.
        # --------------------------------------------------

        tissue_ratio = self._get_slide_tissue_ratio(
            slide_id
        )

        if tissue_ratio is not None:
            slide["tissue_ratio"] = tissue_ratio

        slide["metadata"] = metadata

        return slide

    # ======================================================
    # TILES
    # ======================================================

    def get_tiles(
        self,
        slide_id: str,
    ) -> list[dict[str, Any]]:
        """
        Return all tiles for a slide.

        This method is intentionally retained for
        compatibility with retrieval and analysis services.
        """

        manifest = self._load_tile_manifest(
            slide_id,
            required=True,
        )

        tiles = manifest.get(
            "tiles",
            [],
        )

        return [
            self._normalize_tile(tile)
            for tile in tiles
        ]

    def list_tiles(
        self,
        slide_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """
        Return paginated tile metadata.
        """

        if page < 1:
            raise ValueError(
                "Page must be >= 1."
            )

        if page_size < 1:
            raise ValueError(
                "Page size must be >= 1."
            )

        tiles = self.get_tiles(
            slide_id
        )

        total_tiles = len(tiles)

        total_pages = (
            (
                total_tiles
                + page_size
                - 1
            )
            // page_size
            if total_tiles > 0
            else 0
        )

        start = (
            (page - 1)
            * page_size
        )

        end = start + page_size

        return {
            "slide_id": slide_id,
            "total_tiles": total_tiles,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "tiles": tiles[start:end],
        }

    # ======================================================
    # THUMBNAIL
    # ======================================================

    def get_thumbnail_path(
        self,
        slide_id: str,
    ) -> Path:
        """
        Return the generated WSI thumbnail.
        """

        candidates = [
            self.wsi_root
            / f"{slide_id}_thumbnail.jpg",

            self.wsi_root
            / f"{slide_id}_thumbnail.jpeg",

            self.wsi_root
            / f"{slide_id}_thumbnail.png",
        ]

        return self._find_existing(
            candidates,
            (
                f"Thumbnail not found for "
                f"slide '{slide_id}'."
            ),
        )

    # ======================================================
    # TISSUE MASK
    # ======================================================

    def get_tissue_mask_path(
        self,
        slide_id: str,
    ) -> Path:
        """
        Return the generated tissue mask.
        """

        candidates = [
            self.wsi_root
            / f"{slide_id}_tissue_mask.png",

            self.wsi_root
            / f"{slide_id}_tissue_mask.jpg",
        ]

        return self._find_existing(
            candidates,
            (
                f"Tissue mask not found for "
                f"slide '{slide_id}'."
            ),
        )

    # ======================================================
    # SLIDE TISSUE RATIO
    # ======================================================

    def _get_slide_tissue_ratio(
        self,
        slide_id: str,
    ) -> float | None:
        """
        Calculate the whole-slide tissue coverage ratio
        from the generated tissue mask.

        This represents the fraction of the entire slide
        covered by detected tissue.

        It is deliberately different from the mean tissue
        ratio of extracted tiles because only tissue-rich
        tiles are retained in the tile manifest.
        """

        try:
            mask_path = self.get_tissue_mask_path(
                slide_id
            )

            with Image.open(mask_path) as image:
                mask = np.asarray(
                    image.convert("L")
                )

            if mask.size == 0:
                return None

            # The tissue mask is binary/grayscale.
            # Any non-zero pixel is treated as tissue.
            tissue_pixels = np.count_nonzero(
                mask
            )

            total_pixels = mask.size

            if total_pixels == 0:
                return None

            ratio = (
                tissue_pixels
                / total_pixels
            )

            return float(ratio)

        except Exception:
            logger.exception(
                "Failed to calculate tissue ratio for slide: %s",
                slide_id,
            )

            return None

    # ======================================================
    # TILE IMAGE
    # ======================================================

    def get_tile_image_path(
        self,
        slide_id: str,
        tile_id: int,
    ) -> Path:
        """
        Resolve an extracted tile using its manifest
        coordinates.
        """

        if tile_id < 0:
            raise ValueError(
                "Tile ID must be non-negative."
            )

        tiles = self.get_tiles(
            slide_id
        )

        tile = next(
            (
                item
                for item in tiles
                if int(
                    item["tile_id"]
                ) == tile_id
            ),
            None,
        )

        if tile is None:
            raise FileNotFoundError(
                (
                    f"Tile '{tile_id}' not found "
                    f"for slide '{slide_id}'."
                )
            )

        x = int(tile["x"])
        y = int(tile["y"])

        candidates = [
            self.tile_root
            / (
                f"{slide_id}_L0_"
                f"X{x:06d}_"
                f"Y{y:06d}.png"
            ),

            self.tile_root
            / (
                f"{slide_id}_L0_"
                f"X{x:06d}_"
                f"Y{y:06d}.jpg"
            ),

            self.tile_root
            / (
                f"{slide_id}_L0_"
                f"X{x:06d}_"
                f"Y{y:06d}.jpeg"
            ),

            self.tile_root
            / f"{slide_id}_tile_{tile_id}.png",

            self.tile_root
            / f"tile_{tile_id}.png",
        ]

        return self._find_existing(
            candidates,
            (
                f"Tile image not found for "
                f"{slide_id}/{tile_id}."
            ),
        )

    # ======================================================
    # MANIFEST
    # ======================================================

    def _load_tile_manifest(
        self,
        slide_id: str,
        required: bool = False,
    ) -> dict[str, Any] | None:
        """
        Load the existing WSI tile manifest.
        """

        candidates = [
            self.wsi_root
            / f"{slide_id}_tiles.json",

            self.wsi_root
            / f"{slide_id}_tile_manifest.json",
        ]

        for path in candidates:
            if path.exists():
                return self._read_json(
                    path
                )

        if required:
            raise FileNotFoundError(
                (
                    f"Tile manifest not found "
                    f"for slide '{slide_id}'."
                )
            )

        return None

    # ======================================================
    # METADATA NORMALIZATION
    # ======================================================

    def _build_slide_metadata(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize WSI metadata into the stable API contract.
        """

        # --------------------------------------------------
        # SLIDE ID
        # --------------------------------------------------

        slide_id = (
            metadata.get("slide_id")
            or metadata.get("id")
            or metadata.get("filename")
            or metadata.get("slide")
        )

        filename = (
            metadata.get("filename")
            or metadata.get("file_name")
            or metadata.get("slide")
            or slide_id
        )

        # --------------------------------------------------
        # DIMENSIONS
        # --------------------------------------------------

        dimensions = metadata.get(
            "dimensions",
            {},
        )

        width = 0
        height = 0

        if isinstance(
            dimensions,
            (list, tuple),
        ):
            if (
                len(dimensions) >= 2
                and self._is_number(
                    dimensions[0]
                )
                and self._is_number(
                    dimensions[1]
                )
            ):
                width = dimensions[0]
                height = dimensions[1]

            elif (
                len(dimensions) > 0
                and isinstance(
                    dimensions[0],
                    (list, tuple),
                )
            ):
                first = dimensions[0]

                if len(first) >= 2:
                    width = first[0]
                    height = first[1]

        elif isinstance(
            dimensions,
            dict,
        ):
            width = dimensions.get(
                "width",
                0,
            )

            height = dimensions.get(
                "height",
                0,
            )

        else:
            width = metadata.get(
                "width",
                0,
            )

            height = metadata.get(
                "height",
                0,
            )

        # --------------------------------------------------
        # WIDTH / HEIGHT DIRECT OVERRIDE
        # --------------------------------------------------

        if metadata.get("width") is not None:
            width = metadata["width"]

        if metadata.get("height") is not None:
            height = metadata["height"]

        # --------------------------------------------------
        # LEVELS
        # --------------------------------------------------

        levels_value = metadata.get(
            "levels",
            1,
        )

        if isinstance(
            levels_value,
            (list, tuple),
        ):
            levels = len(levels_value)

            if levels == 0:
                levels = 1

        elif isinstance(
            levels_value,
            dict,
        ):
            levels = levels_value.get(
                "count",
                levels_value.get(
                    "num_levels",
                    1,
                ),
            )

        else:
            levels = levels_value

        try:
            levels = max(
                int(levels),
                1,
            )
        except (
            TypeError,
            ValueError,
        ):
            levels = 1

        # --------------------------------------------------
        # MPP
        # --------------------------------------------------

        mpp = metadata.get(
            "mpp",
            {},
        )

        mpp_x = None
        mpp_y = None

        if isinstance(
            mpp,
            dict,
        ):
            mpp_x = metadata.get(
                "mpp_x",
                mpp.get("x"),
            )

            mpp_y = metadata.get(
                "mpp_y",
                mpp.get("y"),
            )

        elif isinstance(
            mpp,
            (list, tuple),
        ):
            if len(mpp) >= 2:
                mpp_x = mpp[0]
                mpp_y = mpp[1]

        else:
            mpp_x = metadata.get(
                "mpp_x"
            )

            mpp_y = metadata.get(
                "mpp_y"
            )

        # --------------------------------------------------
        # OBJECTIVE
        # --------------------------------------------------

        objective = (
            metadata.get(
                "objective_power"
            )
            or metadata.get(
                "objective"
            )
            or metadata.get(
                "Objective"
            )
        )

        # --------------------------------------------------
        # VENDOR / FORMAT
        # --------------------------------------------------

        vendor = (
            metadata.get("vendor")
            or metadata.get("Vendor")
        )

        slide_format = (
            metadata.get("format")
            or metadata.get("Format")
        )

        # --------------------------------------------------
        # TISSUE RATIO
        # --------------------------------------------------

        tissue_ratio = metadata.get(
            "tissue_ratio"
        )

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        return {
            "slide_id": str(slide_id),
            "filename": str(filename),
            "format": slide_format,
            "vendor": vendor,
            "width": int(width or 0),
            "height": int(height or 0),
            "levels": levels,
            "mpp_x": (
                float(mpp_x)
                if mpp_x is not None
                else None
            ),
            "mpp_y": (
                float(mpp_y)
                if mpp_y is not None
                else None
            ),
            "objective_power": (
                float(objective)
                if objective is not None
                else None
            ),
            "tissue_ratio": (
                float(tissue_ratio)
                if tissue_ratio is not None
                else None
            ),
            "tile_count": None,
        }

    # ======================================================
    # TILE NORMALIZATION
    # ======================================================

    @staticmethod
    def _normalize_tile(
        tile: dict[str, Any],
    ) -> dict[str, Any]:

        tile_id = tile.get(
            "tile_id",
            tile.get("id"),
        )

        width = tile.get(
            "width",
            tile.get("w"),
        )

        height = tile.get(
            "height",
            tile.get("h"),
        )

        return {
            "tile_id": int(tile_id),
            "x": int(tile.get("x", 0)),
            "y": int(tile.get("y", 0)),
            "width": int(width),
            "height": int(height),
            "tissue_ratio": (
                float(
                    tile["tissue_ratio"]
                )
                if tile.get(
                    "tissue_ratio"
                ) is not None
                else None
            ),
        }

    # ======================================================
    # HELPERS
    # ======================================================

    @staticmethod
    def _is_number(
        value: Any,
    ) -> bool:
        return isinstance(
            value,
            (int, float),
        )

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict[str, Any]:

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(handle)

    @staticmethod
    def _find_existing(
        candidates: list[Path],
        error_message: str,
    ) -> Path:

        for path in candidates:
            if (
                path.exists()
                and path.is_file()
            ):
                return path

        raise FileNotFoundError(
            error_message
        )


# ==========================================================
# SHARED SERVICE INSTANCE
# ==========================================================

wsi_service = WSIService()