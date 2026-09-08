from __future__ import annotations

from typing import Iterable

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from pathoverse.models.base import BaseModelAdapter, ModelInfo


class HFVisionAdapter(BaseModelAdapter):
    """
    Generic Hugging Face vision encoder adapter.

    The adapter extracts the CLS token from the final
    hidden representation as the image embedding.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        batch_size: int = 8,
        input_size: int = 224,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
        )

        self.input_size = input_size
        self.trust_remote_code = trust_remote_code
        self.processor = None

    def load(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, images: Iterable[Image.Image]) -> dict:
        self.ensure_loaded()

        return self.processor(
            images=list(images),
            return_tensors="pt",
        )

    @torch.inference_mode()
    def encode(self, images) -> torch.Tensor:
        self.ensure_loaded()

        inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in images.items()
        }

        outputs = self.model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output

        elif hasattr(outputs, "last_hidden_state"):
            embeddings = outputs.last_hidden_state[:, 0]

        else:
            raise RuntimeError(
                "Unable to determine image embedding from model output."
            )

        return embeddings.float()

    def info(self) -> ModelInfo:
        embedding_dim = None

        if self.model is not None:
            config = self.model.config

            embedding_dim = getattr(
                config,
                "hidden_size",
                None,
            )

        return ModelInfo(
            name=self.model_id.split("/")[-1],
            model_id=self.model_id,
            embedding_dim=embedding_dim,
            input_size=self.input_size,
            modality="histopathology",
            description="Generic Hugging Face vision encoder",
        )