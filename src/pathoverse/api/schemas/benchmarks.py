from typing import Optional

from pydantic import BaseModel


class BenchmarkRecord(BaseModel):
    model: str
    model_id: Optional[str] = None
    task: str
    dataset: Optional[str] = None
    split: Optional[str] = None
    samples: Optional[int] = None

    embedding_dimension: Optional[int] = None

    accuracy: Optional[float] = None
    auroc: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    recall_at_1: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None

    latency_ms: Optional[float] = None
    throughput: Optional[float] = None
    gpu_memory_mb: Optional[float] = None

    status: str = "completed"


class BenchmarkResponse(BaseModel):
    total_records: int
    records: list[BenchmarkRecord]
