from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from timm.data import (
    create_transform,
    resolve_data_config,
)

import gigapath.tile_encoder as tile_encoder

from ..base import BaseModelAdapter, ModelInfo


class GigaPathFlashAdapter(
    BaseModelAdapter
):

    def __init__(
        self,
        model_id=(
            "hf_hub:"
            "prov-gigapath/"
            "prov-gigapath-flash"
        ),
        device="cuda",
        batch_size=8,
    ):

        super().__init__(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
        )

        self.model = None
        self.transform = None

        # GigaPath-Flash fixed architecture metadata.
        self.embedding_dim = 384
        self.input_size = 224

    # ========================================================
    # LOAD
    # ========================================================

    def load(self):

        self.model = (
            tile_encoder.create_model(
                self.model_id
            )
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

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

            raise RuntimeError(
                f"Expected 2D embeddings, "
                f"got {embeddings.shape}"
            )

        if embeddings.shape[1] != self.embedding_dim:

            raise RuntimeError(
                "Unexpected GigaPath-Flash "
                f"embedding dimension: "
                f"{embeddings.shape[1]}. "
                f"Expected {self.embedding_dim}."
            )

        return embeddings.float()

    # ========================================================
    # INFO
    # ========================================================

    def info(self):

        return ModelInfo(
            name="GigaPath-Flash",
            model_id=self.model_id,
            embedding_dim=self.embedding_dim,
            input_size=self.input_size,
            modality="histopathology",
            description=(
                "GigaPath-Flash pathology "
                "tile encoder."
            ),
        )