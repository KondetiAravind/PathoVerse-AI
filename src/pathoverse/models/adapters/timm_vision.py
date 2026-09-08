from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from timm import create_model
from timm.data import (
    create_transform,
    resolve_data_config,
)

from ..base import BaseModelAdapter, ModelInfo


class TimmVisionAdapter(BaseModelAdapter):

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        batch_size: int = 8,
        pretrained: bool = True,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
        )

        self.pretrained = pretrained

        self.model = None
        self.transform = None

        # Known metadata for ViT-B/16.
        # This allows info() to work before load().
        self.embedding_dim = 768
        self.input_size = 224

    # ========================================================
    # LOAD
    # ========================================================

    def load(self):

        self.model = create_model(
            self.model_id,
            pretrained=self.pretrained,
            num_classes=0,
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        # Resolve actual model metadata after loading.
        self.embedding_dim = int(
            self.model.num_features
        )

        config = resolve_data_config(
            {},
            model=self.model,
        )

        self.transform = create_transform(
            **config,
            is_training=False,
        )

    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(
        self,
        images: Sequence[Image.Image],
    ):

        if self.transform is None:

            raise RuntimeError(
                "Model must be loaded before "
                "preprocessing images."
            )

        tensors = []

        for image in images:

            if not isinstance(
                image,
                Image.Image,
            ):

                raise TypeError(
                    "Expected PIL.Image."
                )

            image = image.convert(
                "RGB"
            )

            tensors.append(
                self.transform(
                    image
                )
            )

        return torch.stack(
            tensors
        )

    # ========================================================
    # ENCODE
    # ========================================================

    @torch.inference_mode()
    def encode(
        self,
        images,
    ):

        self.ensure_loaded()

        tensors = self.preprocess(
            images
        ).to(
            self.device
        )

        embeddings = self.model(
            tensors
        )

        if embeddings.ndim != 2:

            embeddings = embeddings.flatten(
                start_dim=1
            )

        return embeddings.float()

    # ========================================================
    # INFO
    # ========================================================

    def info(self):

        return ModelInfo(
            name="ViT-B/16",
            model_id=self.model_id,
            embedding_dim=self.embedding_dim,
            input_size=self.input_size,
            modality="histopathology",
            description=(
                "ViT-B/16 image foundation "
                "encoder."
            ),
        )