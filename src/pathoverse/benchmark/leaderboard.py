from __future__ import annotations

from typing import Any


# ============================================================
# MODEL SUMMARY
# ============================================================

def build_model_summary(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Combine all benchmark records belonging to the
    same foundation model.
    """

    models: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in records:

        model_id = record["model_id"]

        if model_id not in models:

            models[model_id] = {
                "model": record["model"],
                "model_id": model_id,
                "embedding_dimension": (
                    record["embedding_dimension"]
                ),
                "tasks": {},
            }

        task = record["task"]

        models[model_id]["tasks"][task] = {
            key: value
            for key, value in record.items()
            if key not in {
                "model",
                "model_id",
                "task",
                "embedding_dimension",
            }
        }

    return list(models.values())


# ============================================================
# CLASSIFICATION RANKING
# ============================================================

def build_classification_ranking(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Rank classification experiments by AUROC.
    """

    rows = [
        dict(record)
        for record in records
        if record.get("task")
        == "classification"
        and record.get("status")
        == "completed"
        and record.get("auroc") is not None
    ]

    rows.sort(
        key=lambda row: float(
            row["auroc"]
        ),
        reverse=True,
    )

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        row["classification_rank"] = rank

    return rows


# ============================================================
# EFFICIENCY RANKING
# ============================================================

def build_efficiency_ranking(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Rank foundation models by throughput.

    Higher throughput = better rank.
    """

    rows = [
        dict(record)
        for record in records
        if record.get("task")
        == "foundation_efficiency"
        and record.get("throughput") is not None
    ]

    rows.sort(
        key=lambda row: float(
            row["throughput"]
        ),
        reverse=True,
    )

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        row["efficiency_rank"] = rank

    return rows


# ============================================================
# RETRIEVAL RANKING
# ============================================================

def build_retrieval_ranking(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Rank retrieval models by Recall@5.
    """

    rows = [
        dict(record)
        for record in records
        if record.get("task")
        == "image_retrieval"
        and record.get("recall_at_5") is not None
    ]

    rows.sort(
        key=lambda row: float(
            row["recall_at_5"]
        ),
        reverse=True,
    )

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        row["retrieval_rank"] = rank

    return rows