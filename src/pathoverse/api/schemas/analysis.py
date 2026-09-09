from typing import Optional

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    model: str
    dataset: str = "patchcamelyon"


class ClassificationResponse(BaseModel):
    model: str
    dataset: str
    accuracy: Optional[float] = None
    auroc: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None


class MILRequest(BaseModel):
    slide_id: str
    model: str = "gigapath-flash"


class MILFoundationModel(BaseModel):
    name: str
    embedding_dimension: int


class MILMetadata(BaseModel):
    architecture: str
    hidden_dimension: int
    attention_dimension: int
    num_classes: int
    trained: bool
    prototype: bool
    seed: Optional[int] = None


class MILPrediction(BaseModel):
    class_id: int
    probability: float


class MILAttention(BaseModel):
    tile_count: int
    sum: float
    top_k: int
    top_tiles: list[dict] = Field(
        default_factory=list
    )


class MILResponse(BaseModel):
    schema_version: str
    task: str
    slide_id: str
    model: str

    foundation_model: MILFoundationModel
    mil: MILMetadata
    prediction: MILPrediction
    attention: MILAttention

    slide_embedding_dimension: int
    slide_embedding_path: str

    trained: bool
    prototype: bool