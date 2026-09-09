from fastapi import APIRouter, HTTPException

from pathoverse.api.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResultItem,
)
from pathoverse.api.services.retrieval_service import (
    retrieval_service,
)


router = APIRouter()


@router.post(
    "/search",
    response_model=RetrievalResponse,
)
def search_similar_tiles(
    request: RetrievalRequest,
):
    try:
        results = retrieval_service.search(
            slide_id=request.slide_id,
            tile_id=request.tile_id,
            model=request.model,
            top_k=request.top_k,
        )

        return RetrievalResponse(
            query_tile_id=request.tile_id,
            model=request.model,
            top_k=len(results),
            results=[
                RetrievalResultItem(**item)
                for item in results
            ],
        )

    except (
        FileNotFoundError,
        IndexError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
