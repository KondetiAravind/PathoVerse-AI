from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    model_id: str
    batch_size: int = 8
    device: str = "cuda"
    input_size: int = 224
    embedding_dim: int | None = None
    trust_remote_code: bool = False
    normalize_embeddings: bool = False


DEFAULT_MODEL_CONFIGS = {
    "vit-base": ModelConfig(
        name="vit-base",
        model_id="google/vit-base-patch16-224-in21k",
        batch_size=8,
        input_size=224,
        embedding_dim=768,
    ),
}