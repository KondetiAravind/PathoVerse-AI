from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from pathoverse.api.schemas.slides import (
    SlideDetail,
    SlideSummary,
    SlideTileResponse,
    TileInfo,
)
from pathoverse.api.services.wsi_service import (
    wsi_service,
)


router = APIRouter()


@router.get(
    "",
    response_model=list[SlideSummary],
)
def list_slides():
    return wsi_service.list_slides()


@router.get(
    "/{slide_id}",
    response_model=SlideDetail,
)
def get_slide(slide_id: str):
    try:
        data = wsi_service.get_slide(
            slide_id
        )

        dimensions = data.get(
            "dimensions",
            [
                data.get("width", 0),
                data.get("height", 0),
            ],
        )

        raw_levels = data.get(
            "level_details",
            [],
        )

        return SlideDetail(
            slide_id=data.get(
                "slide_id",
                slide_id,
            ),
            filename=data.get(
                "filename",
                f"{slide_id}.svs",
            ),
            format=data.get("format"),
            vendor=data.get("vendor"),
            width=dimensions[0],
            height=dimensions[1],
            levels=data.get(
                "level_count",
                data.get(
                    "levels",
                    len(raw_levels),
                ),
            ),
            mpp_x=data.get("mpp_x"),
            mpp_y=data.get("mpp_y"),
            objective_power=data.get(
                "objective_power"
            ),
            dimensions=dimensions,
            level_details=raw_levels,
            properties=data.get(
                "properties",
                {},
            ),
            tissue_ratio=data.get(
                "tissue_ratio"
            ),
            tile_count=data.get(
                "tile_count"
            ),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@router.get(
    "/{slide_id}/thumbnail",
)
def get_thumbnail(slide_id: str):
    try:
        return FileResponse(
            wsi_service.get_thumbnail_path(
                slide_id
            ),
            media_type="image/jpeg",
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@router.get(
    "/{slide_id}/tissue-mask",
)
def get_tissue_mask(slide_id: str):
    try:
        return FileResponse(
            wsi_service.get_mask_path(
                slide_id
            ),
            media_type="image/png",
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@router.get(
    "/{slide_id}/tiles",
    response_model=SlideTileResponse,
)
def get_slide_tiles(slide_id: str):
    try:
        tiles = wsi_service.get_tiles(
            slide_id
        )

        response_tiles = [
            TileInfo(
                tile_id=int(item["tile_id"]),
                slide_id=slide_id,
                x=int(item["x"]),
                y=int(item["y"]),
                width=int(item["width"]),
                height=int(item["height"]),
                level=0,
                tissue_ratio=float(
                    item["tissue_ratio"]
                ),
            )
            for item in tiles
        ]

        return SlideTileResponse(
            slide_id=slide_id,
            total_tiles=len(response_tiles),
            tiles=response_tiles,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@router.get(
    "/{slide_id}/tiles/{tile_id}/image",
)
def get_tile_image(
    slide_id: str,
    tile_id: int,
):
    try:
        image_bytes = (
            wsi_service.get_tile_image(
                slide_id,
                tile_id,
            )
        )

        return StreamingResponse(
            iter([image_bytes]),
            media_type="image/png",
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except IndexError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc