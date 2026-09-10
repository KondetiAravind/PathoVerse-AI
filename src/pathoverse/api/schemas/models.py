from typing import Optional

from pydantic import BaseModel


class ModelSummary(BaseModel):
    name: str
    model_id: str
    embedding_dimension: Optional[int] = None
    input_size: int
    modality: str
    description: str = ""
    status: str = "available"


class ModelBenchmark(BaseModel):
    model: str
    task: str
    dataset: Optional[str] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    embedding_dimension: Optional[int] = None
    latency_ms: Optional[float] = None
    throughput: Optional[float] = None
    gpu_memory_mb: Optional[float] = None