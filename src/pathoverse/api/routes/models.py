from fastapi import APIRouter, HTTPException

from pathoverse.api.schemas.models import ModelSummary
from pathoverse.api.services.model_service import (
    model_service,
)


router = APIRouter()


# ============================================================
# MODEL CATALOG
# ============================================================


@router.get(
    "",
    response_model=list[ModelSummary],
)
def list_models():

    return [
        ModelSummary(
            name=model.name,
            model_id=model.key,
            embedding_dimension=(
                model.embedding_dimension
            ),
            input_size=model.input_size,
            modality=model.modality,
            description=model.description,
            status=(
                "loaded"
                if model_service.is_loaded(
                    model.key
                )
                else "available"
            ),
        )
        for model in model_service.list_models()
    ]


# ============================================================
# SYSTEM STATUS
# ============================================================


@router.get(
    "/status",
)
def model_system_status():

    return model_service.system_status()


# ============================================================
# UNLOAD ALL
#
# This static route is intentionally declared before the
# parameterized route.
# ============================================================


@router.post(
    "/unload-all",
)
def unload_all_models():

    model_service.unload_all()

    return model_service.system_status()


# ============================================================
# MODEL DETAIL
# ============================================================


@router.get(
    "/{model_id}",
)
def get_model(
    model_id: str,
):

    try:

        return model_service.status(
            model_id
        )

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


# ============================================================
# LOAD MODEL
# ============================================================


@router.post(
    "/{model_id}/load",
)
def load_model(
    model_id: str,
):

    try:

        definition = (
            model_service.get_definition(
                model_id
            )
        )

        model_service.load(
            definition.key
        )

        return model_service.status(
            definition.key
        )

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to load model "
                f"'{model_id}': {exc}"
            ),
        )


# ============================================================
# UNLOAD MODEL
# ============================================================


@router.post(
    "/{model_id}/unload",
)
def unload_model(
    model_id: str,
):

    try:

        definition = (
            model_service.get_definition(
                model_id
            )
        )

        model_service.unload(
            definition.key
        )

        return model_service.status(
            definition.key
        )

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to unload model "
                f"'{model_id}': {exc}"
            ),
        )