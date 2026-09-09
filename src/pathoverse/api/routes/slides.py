from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from pathoverse.api.schemas.slides import (
    SlideResponse,
    TileListResponse,
)
from pathoverse.api.services.wsi_service import (
    wsi_service,
)


router = APIRouter()

logger = logging.getLogger(
    "pathoverse.api.slides"
)


# ==========================================================
# SLIDE LIST
# ==========================================================

@router.get(
    "",
    summary="List available whole-slide images",
)
def list_slides():
    """
    Return all available WSI slides.

    The response intentionally remains a raw list to
    preserve the existing API contract.
    """

    try:
        return wsi_service.list_slides()

    except Exception as exc:
        logger.exception(
            "Failed to list slides"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to list "
                "whole-slide images."
            ),
        ) from exc


# ==========================================================
# SLIDE DETAIL
# ==========================================================

@router.get(
    "/{slide_id}",
    response_model=SlideResponse,
    summary="Get whole-slide image metadata",
)
def get_slide(
    slide_id: str,
):
    """
    Return detailed metadata for a WSI.
    """

    try:
        return wsi_service.get_slide(
            slide_id
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "SLIDE_NOT_FOUND",
                "message": (
                    f"Slide '{slide_id}' "
                    "was not found."
                ),
            },
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to retrieve slide: %s",
            slide_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load "
                "slide metadata."
            ),
        ) from exc


# ==========================================================
# THUMBNAIL
# ==========================================================

@router.get(
    "/{slide_id}/thumbnail",
    summary="Get WSI thumbnail",
)
def get_thumbnail(
    slide_id: str,
):
    """
    Return the generated WSI thumbnail.
    """

    try:
        path = (
            wsi_service
            .get_thumbnail_path(slide_id)
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "THUMBNAIL_NOT_FOUND",
                "message": (
                    f"Thumbnail for slide "
                    f"'{slide_id}' was not found."
                ),
            },
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to retrieve thumbnail: %s",
            slide_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "slide thumbnail."
            ),
        ) from exc

    media_type = (
        "image/jpeg"
        if path.suffix.lower()
        in {".jpg", ".jpeg"}
        else "image/png"
    )

    return FileResponse(
        path=path,
        media_type=media_type,
        filename=path.name,
    )


# ==========================================================
# TISSUE MASK
# ==========================================================

@router.get(
    "/{slide_id}/tissue-mask",
    summary="Get WSI tissue mask",
)
def get_tissue_mask(
    slide_id: str,
):
    """
    Return the generated tissue mask.
    """

    try:
        path = (
            wsi_service
            .get_tissue_mask_path(slide_id)
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "TISSUE_MASK_NOT_FOUND",
                "message": (
                    f"Tissue mask for slide "
                    f"'{slide_id}' was not found."
                ),
            },
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to retrieve tissue mask: %s",
            slide_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "tissue mask."
            ),
        ) from exc

    return FileResponse(
        path=path,
        media_type="image/png",
        filename=path.name,
    )


# ==========================================================
# TILE LIST
# ==========================================================

@router.get(
    "/{slide_id}/tiles",
    response_model=TileListResponse,
    summary="List extracted WSI tiles",
)
def list_tiles(
    slide_id: str,
    page: int = Query(
        default=1,
        ge=1,
        description="1-based page number.",
    ),
    page_size: int = Query(
        default=50,
        ge=1,
        le=500,
        description=(
            "Number of tiles returned "
            "per page."
        ),
    ),
):
    """
    Return paginated tile metadata.

    The response preserves the existing
    `total_tiles` field.
    """

    try:
        return wsi_service.list_tiles(
            slide_id=slide_id,
            page=page,
            page_size=page_size,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": (
                    "TILE_MANIFEST_NOT_FOUND"
                ),
                "message": (
                    f"Tile manifest for slide "
                    f"'{slide_id}' was not found."
                ),
            },
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_TILE_QUERY",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to retrieve tiles: %s",
            slide_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "slide tiles."
            ),
        ) from exc


# ==========================================================
# TILE IMAGE
# ==========================================================

@router.get(
    "/{slide_id}/tiles/{tile_id}/image",
    summary="Get extracted tile image",
)
def get_tile_image(
    slide_id: str,
    tile_id: int,
):
    """
    Return one extracted tile image.
    """

    if tile_id < 0:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_TILE_ID",
                "message": (
                    "Tile ID must be "
                    "non-negative."
                ),
            },
        )

    try:
        path = (
            wsi_service
            .get_tile_image_path(
                slide_id=slide_id,
                tile_id=tile_id,
            )
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "TILE_NOT_FOUND",
                "message": (
                    f"Tile '{tile_id}' for "
                    f"slide '{slide_id}' "
                    "was not found."
                ),
            },
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_TILE_ID",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to retrieve tile image: "
            "%s/%s",
            slide_id,
            tile_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to retrieve "
                "tile image."
            ),
        ) from exc

    media_type = (
        "image/jpeg"
        if path.suffix.lower()
        in {".jpg", ".jpeg"}
        else "image/png"
    )

    return FileResponse(
        path=path,
        media_type=media_type,
        filename=Path(path).name,
    )