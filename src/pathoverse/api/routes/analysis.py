from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from pathoverse.api.schemas.analysis import (
    ClassificationRequest,
    ClassificationResponse,
    MILAttention,
    MILFoundationModel,
    MILMetadata,
    MILPrediction,
    MILRequest,
    MILResponse,
)
from pathoverse.api.services.analysis_service import (
    analysis_service,
)
from pathoverse.api.services.model_service import (
    normalize_model_id,
)


router = APIRouter()


@router.post(
    "/classification",
    response_model=ClassificationResponse,
)
def classification(
    request: ClassificationRequest,
):
    model = normalize_model_id(request.model)

    try:
        data = analysis_service.get_classification(
            model
        )

        model_data = data.get("model", {})
        test_data = data.get("test", {})

        return ClassificationResponse(
            model=model_data.get(
                "name",
                model,
            ),
            dataset=request.dataset,
            accuracy=test_data.get("accuracy"),
            auroc=test_data.get("auroc"),
            f1=test_data.get("f1"),
            precision=test_data.get("precision"),
            recall=test_data.get("recall"),
            sensitivity=test_data.get(
                "sensitivity"
            ),
            specificity=test_data.get(
                "specificity"
            ),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@router.post(
    "/mil",
    response_model=MILResponse,
)
def mil_analysis(
    request: MILRequest,
):
    model = normalize_model_id(request.model)

    try:
        data = analysis_service.get_mil(
            request.slide_id,
            model,
        )

        foundation = data.get(
            "foundation_model",
            {},
        )

        mil = data.get(
            "mil",
            {},
        )

        prediction = data.get(
            "prediction",
            {},
        )

        embedding = data.get(
            "slide_embedding",
            {},
        )

        attention = data.get(
            "attention",
            {},
        )

        return MILResponse(
            schema_version=data.get(
                "schema_version",
                "1.0",
            ),
            task=data.get(
                "task",
                "wsi_attention_mil_prototype",
            ),
            slide_id=data.get(
                "slide_id",
                request.slide_id,
            ),
            model=model,
            foundation_model=MILFoundationModel(
                name=foundation.get(
                    "name",
                    model,
                ),
                embedding_dimension=foundation.get(
                    "embedding_dimension",
                    0,
                ),
            ),
            mil=MILMetadata(
                architecture=mil.get(
                    "architecture",
                    "unknown",
                ),
                hidden_dimension=mil.get(
                    "hidden_dimension",
                    0,
                ),
                attention_dimension=mil.get(
                    "attention_dimension",
                    0,
                ),
                num_classes=mil.get(
                    "num_classes",
                    0,
                ),
                trained=mil.get(
                    "trained",
                    False,
                ),
                prototype=mil.get(
                    "prototype",
                    True,
                ),
                seed=mil.get("seed"),
            ),
            prediction=MILPrediction(
                class_id=prediction.get(
                    "class",
                    0,
                ),
                probability=prediction.get(
                    "probability",
                    0.0,
                ),
            ),
            attention=MILAttention(
                tile_count=attention.get(
                    "tile_count",
                    0,
                ),
                sum=attention.get(
                    "sum",
                    0.0,
                ),
                top_k=attention.get(
                    "top_k",
                    0,
                ),
                top_tiles=attention.get(
                    "top_tiles",
                    [],
                ),
            ),
            slide_embedding_dimension=embedding.get(
                "dimension",
                0,
            ),
            slide_embedding_path=(
                Path(str(embedding.get("path", ""))).name
                if embedding.get("path")
                else ""
            ),
            trained=mil.get(
                "trained",
                False,
            ),
            prototype=mil.get(
                "prototype",
                True,
            ),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@router.get(
    "/{slide_id}/heatmap",
)
def get_heatmap(
    slide_id: str,
    model: str = "gigapath-flash",
):
    model = normalize_model_id(model)

    try:
        path = analysis_service.get_heatmap_path(
            slide_id,
            model,
        )

        return FileResponse(
            path,
            media_type="image/png",
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc