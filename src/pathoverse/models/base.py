from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import torch


@dataclass(frozen=True)
class ModelInfo:
    name: str
    model_id: str
    embedding_dim: int | None
    input_size: int
    modality: str = "histopathology"
    description: str = ""


class BaseModelAdapter(ABC):
    """
    Common interface for all PathoVerse foundation models.

    Every model adapter must implement:
        - load()
        - preprocess()
        - encode()
        - info()
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = batch_size
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights and preprocessing configuration."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, images: Iterable[Any]) -> Any:
        """Convert raw images into model-ready tensors."""
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode()
    def encode(self, images: Any) -> torch.Tensor:
        """Generate embeddings."""
        raise NotImplementedError

    @abstractmethod
    def info(self) -> ModelInfo:
        """Return model metadata."""
        raise NotImplementedError

    def ensure_loaded(self) -> None:
        if self.model is None:
            self.load()

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)