from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ==========================================================
# SLIDE METADATA
# ==========================================================

class SlideMetadata(BaseModel):
    slide_id: str

    filename: str

    format: str | None = None

    vendor: str | None = None

    width: int = Field(ge=0)

    height: int = Field(ge=0)

    levels: int = Field(ge=1)

    mpp_x: float | None = Field(
        default=None,
        ge=0,
    )

    mpp_y: float | None = Field(
        default=None,
        ge=0,
    )

    objective_power: float | None = Field(
        default=None,
        ge=0,
    )

    tissue_ratio: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    tile_count: int | None = Field(
        default=None,
        ge=0,
    )


# ==========================================================
# TILE METADATA
# ==========================================================

class TileMetadata(BaseModel):
    tile_id: int = Field(ge=0)

    x: int = Field(ge=0)

    y: int = Field(ge=0)

    width: int = Field(gt=0)

    height: int = Field(gt=0)

    tissue_ratio: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


# ==========================================================
# TILE LIST RESPONSE
# ==========================================================

class TileListResponse(BaseModel):
    slide_id: str

    total_tiles: int = Field(
        ge=0
    )

    page: int = Field(
        ge=1
    )

    page_size: int = Field(
        ge=1
    )

    total_pages: int = Field(
        ge=0
    )

    tiles: list[TileMetadata]


# ==========================================================
# SLIDE DETAIL
# ==========================================================

class SlideResponse(SlideMetadata):

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    tiling: dict[str, Any] = Field(
        default_factory=dict
    )

    statistics: dict[str, Any] = Field(
        default_factory=dict
    )