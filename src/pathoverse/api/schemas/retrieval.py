from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    slide_id: str
    tile_id: int
    model: str = "gigapath-flash"
    top_k: int = Field(default=5, ge=1, le=100)


class RetrievalResultItem(BaseModel):
    tile_id: int
    slide_id: str
    score: float
    x: int
    y: int
    width: int
    height: int


class RetrievalResponse(BaseModel):
    query_tile_id: int
    model: str
    top_k: int
    results: list[RetrievalResultItem]
