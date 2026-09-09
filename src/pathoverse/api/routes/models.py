from fastapi import APIRouter, HTTPException

from pathoverse.api.schemas.models import ModelSummary
from pathoverse.api.services.model_service import (
    model_service,
)


router = APIRouter()


@router.get(
    "",
    response_model=list[ModelSummary],
)
def list_models():
    return [
        ModelSummary(
            name=model.name,
            model_id=model.key,
            embedding_dimension=model.embedding_dimension,
            input_size=model.input_size,
            modality=model.modality,
            description=model.description,
            status="loaded"
            if model_service.is_loaded(model.key)
            else "available",
        )
        for model in model_service.list_models()
    ]


@router.get("/{model_id}")
def get_model(model_id: str):
    try:
        return model_service.status(
            model_id
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
