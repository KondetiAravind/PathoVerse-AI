from __future__ import annotations

from typing import Iterable

import torch
from PIL import Image

from pathoverse.models.registry import registry


class ModelEngine:
    """
    High-level inference interface for PathoVerse foundation models.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.adapter = registry.create(
            model_name,
            device=device,
            batch_size=batch_size,
        )

    def load(self) -> None:
        self.adapter.load()

    def encode_images(
        self,
        images: Iterable[Image.Image],
    ) -> torch.Tensor:
        images = list(images)

        if not images:
            raise ValueError("No images supplied.")

        outputs = []

        for start in range(
            0,
            len(images),
            self.batch_size,
        ):
            batch = images[start : start + self.batch_size]

            inputs = self.adapter.preprocess(batch)

            embeddings = self.adapter.encode(inputs)

            outputs.append(
                embeddings.detach().cpu()
            )

        return torch.cat(outputs, dim=0)

    def info(self):
        return self.adapter.info()