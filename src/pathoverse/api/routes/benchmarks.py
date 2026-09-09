from fastapi import APIRouter, HTTPException

from pathoverse.api.schemas.benchmarks import (
    BenchmarkRecord,
    BenchmarkResponse,
)
from pathoverse.api.services.benchmark_service import (
    benchmark_service,
)
from pathoverse.api.services.model_service import (
    normalize_model_id,
)


router = APIRouter()

def _convert(record: dict) -> BenchmarkRecord:
    raw_model_id = record.get(
        "model_id"
    )

    raw_model = record.get(
        "model",
        "",
    )

    model_id = normalize_model_id(
        raw_model_id or raw_model
    )

    return BenchmarkRecord(
        model=raw_model,
        model_id=model_id,
        task=record.get("task", ""),
        dataset=record.get("dataset"),
        split=record.get("split"),
        samples=record.get("samples"),
        embedding_dimension=record.get(
            "embedding_dimension"
        ),
        accuracy=record.get("accuracy"),
        auroc=record.get("auroc"),
        f1=record.get("f1"),
        precision=record.get("precision"),
        recall=record.get("recall"),
        recall_at_1=record.get(
            "recall_at_1"
        ),
        recall_at_5=record.get(
            "recall_at_5"
        ),
        recall_at_10=record.get(
            "recall_at_10"
        ),
        latency_ms=record.get(
            "latency_ms"
        ),
        throughput=record.get(
            "throughput"
        ),
        gpu_memory_mb=record.get(
            "gpu_memory_mb"
        ),
        status=record.get(
            "status",
            "completed",
        ),
    )

@router.get(
    "",
    response_model=BenchmarkResponse,
)
def list_benchmarks():
    try:
        records = benchmark_service.records()

        return BenchmarkResponse(
            total_records=len(records),
            records=[
                _convert(record)
                for record in records
            ],
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


@router.get(
    "/{task}",
    response_model=BenchmarkResponse,
)
def get_benchmark_task(task: str):
    try:
        records = benchmark_service.by_task(
            task
        )

        if not records:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No benchmark records "
                    f"found for task '{task}'."
                ),
            )

        return BenchmarkResponse(
            total_records=len(records),
            records=[
                _convert(record)
                for record in records
            ],
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
