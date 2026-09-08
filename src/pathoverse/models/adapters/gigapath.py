from __future__ import annotations

from typing import Iterable

import torch
from PIL import Image

import gigapath.tile_encoder as tile_encoder
from timm.data import create_transform, resolve_data_config

from pathoverse.models.base import BaseModelAdapter, ModelInfo


class GigaPathFlashAdapter(BaseModelAdapter):
    """
    PathoVerse adapter for the official Prov-GigaPath-Flash
    pathology tile encoder.

    Model:
        prov-gigapath/prov-gigapath-flash

    Architecture:
        gigapath_tile_enc_dinov2s

    Input:
        224 x 224 RGB

    Output:
        384-dimensional tile embedding
    """

    def __init__(
        self,
        model_id: str = "hf_hub:prov-gigapath/prov-gigapath-flash",
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:

        super().__init__(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
        )

        self.input_size = 224
        self.embedding_dim = 384
        self.transform = None

    # ==============================================================
    # LOAD
    # ==============================================================

    def load(self) -> None:
        """
        Load the official GigaPath-Flash tile encoder.
        """

        self.model = tile_encoder.create_model(
            self.model_id
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # The official GigaPath model exposes its preprocessing
        # configuration through TIMM's default_cfg.
        data_config = resolve_data_config(
            {},
            model=self.model,
        )

        self.transform = create_transform(
            **data_config,
            is_training=False,
        )

    # ==============================================================
    # PREPROCESS
    # ==============================================================

    def preprocess(
        self,
        images: Iterable[Image.Image],
    ) -> torch.Tensor:
        """
        Convert PIL pathology images into model-ready tensors.

        Returns:
            Tensor [N, 3, 224, 224]
        """

        self.ensure_loaded()

        images = list(images)

        if not images:
            raise ValueError(
                "images cannot be empty"
            )

        tensors = []

        for image in images:

            if not isinstance(
                image,
                Image.Image,
            ):
                raise TypeError(
                    "All inputs must be PIL.Image.Image objects."
                )

            image = image.convert("RGB")

            tensors.append(
                self.transform(image)
            )

        return torch.stack(tensors)

    # ==============================================================
    # ENCODE
    # ==============================================================

    @torch.inference_mode()
    def encode(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate GigaPath-Flash tile embeddings.

        Args:
            images:
                Preprocessed tensor [N, 3, 224, 224].

        Returns:
            Tensor [N, 384].
        """

        self.ensure_loaded()

        images = self.to_device(images)

        embeddings = self.model(
            images
        )

        if embeddings.ndim != 2:
            raise RuntimeError(
                "Unexpected GigaPath embedding shape: "
                f"{tuple(embeddings.shape)}"
            )

        if embeddings.shape[1] != self.embedding_dim:
            raise RuntimeError(
                "Unexpected GigaPath embedding dimension: "
                f"{embeddings.shape[1]} "
                f"(expected {self.embedding_dim})"
            )

        return embeddings.float()

    # ==============================================================
    # INFO
    # ==============================================================

    def info(self) -> ModelInfo:
        """
        Return standardized PathoVerse model metadata.
        """

        return ModelInfo(
            name="GigaPath-Flash",
            model_id=self.model_id,
            embedding_dim=self.embedding_dim,
            input_size=self.input_size,
            modality="histopathology",
            description=(
                "Prov-GigaPath-Flash pathology tile encoder "
                "using a DINOv2-Small architecture"
            ),
        )