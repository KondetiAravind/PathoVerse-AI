from pydantic import BaseModel


class SlideSummary(BaseModel):
    slide_id: str
    filename: str
    width: int
    height: int
    levels: int
    mpp_x: float | None = None
    mpp_y: float | None = None
    tissue_ratio: float | None = None
    tile_count: int | None = None


class SlideDetail(BaseModel):
    slide_id: str
    filename: str
    format: str | None = None
    vendor: str | None = None

    width: int
    height: int
    levels: int

    mpp_x: float | None = None
    mpp_y: float | None = None
    objective_power: float | None = None

    dimensions: list[int]
    level_details: list[dict]
    properties: dict

    tissue_ratio: float | None = None
    tile_count: int | None = None


class TileInfo(BaseModel):
    tile_id: int
    slide_id: str

    x: int
    y: int
    width: int
    height: int

    level: int
    tissue_ratio: float


class SlideTileResponse(BaseModel):
    slide_id: str
    total_tiles: int
    tiles: list[TileInfo]