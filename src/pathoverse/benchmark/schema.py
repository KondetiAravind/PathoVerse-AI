from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkRecord:
    """
    Standardized PathoVerse benchmark record.

    A record represents one model evaluated on one task.
    """

    model: str
    model_id: str
    task: str

    embedding_dimension: int | None = None

    dataset: str | None = None
    split: str | None = None
    samples: int | None = None

    accuracy: float | None = None
    auroc: float | None = None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None

    recall_at_1: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None

    latency_ms: float | None = None
    throughput: float | None = None
    gpu_memory_mb: float | None = None

    inference_time_sec: float | None = None

    status: str = "completed"

    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert record to a JSON-compatible dictionary.
        """

        return asdict(self)