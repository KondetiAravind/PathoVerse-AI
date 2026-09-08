from __future__ import annotations

from typing import Iterable

import torch
import timm
from PIL import Image
from timm.data import create_transform, resolve_data_config

from pathoverse.models.base import BaseModelAdapter, ModelInfo


class TimmVisionAdapter(BaseModelAdapter):
    """
    Generic adapter for TIMM-based vision foundation models.

    Interface:
        load()
        preprocess()
        encode()
        info()

    Designed for pathology foundation models such as:
        - GigaPath
        - GigaPath-Flash
        - UNI / UNI2
        - other TIMM-compatible encoders
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        batch_size: int = 8,
        pretrained: bool = True,
    ) -> None:
        super().__init__(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
        )

        self.pretrained = pretrained
        self.transform = None
        self.input_size = 224
        self.embedding_dim = None

    # ==============================================================
    # MODEL LOADING
    # ==============================================================

    def load(self) -> None:
        """
        Load the TIMM model and construct its preprocessing pipeline.
        """

        self.model = timm.create_model(
            self.model_id,
            pretrained=self.pretrained,
            num_classes=0,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Resolve preprocessing configuration from the model.
        data_config = resolve_data_config(
            {},
            model=self.model,
        )

        self.transform = create_transform(
            **data_config,
            is_training=False,
        )

        # Determine embedding dimension.
        self.embedding_dim = self._get_embedding_dim()

        # Determine expected input size.
        self.input_size = self._get_input_size()

    # ==============================================================
    # PREPROCESSING
    # ==============================================================

    def preprocess(
        self,
        images: Iterable[Image.Image],
    ) -> torch.Tensor:
        """
        Convert PIL images into a batched tensor.

        Returns:
            Tensor of shape [N, 3, H, W].
        """

        self.ensure_loaded()

        images = list(images)

        if not images:
            raise ValueError("images cannot be empty")

        processed = []

        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError(
                    "All inputs must be PIL.Image.Image objects."
                )

            image = image.convert("RGB")

            processed.append(
                self.transform(image)
            )

        batch = torch.stack(processed)

        return batch

    # ==============================================================
    # ENCODING
    # ==============================================================

    @torch.inference_mode()
    def encode(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate image embeddings.

        Args:
            images:
                Preprocessed tensor [N, 3, H, W].

        Returns:
            Float tensor [N, embedding_dim].
        """

        self.ensure_loaded()

        images = self.to_device(images)

        features = self.model.forward_features(images)

        embeddings = self._pool_features(features)

        return embeddings.float()

    # ==============================================================
    # FEATURE POOLING
    # ==============================================================

    def _pool_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert TIMM feature output into one vector per image.

        Supported representations:

        [B, C]
            Already pooled.

        [B, N, C]
            Token representation.
            Uses the first token (CLS/token pooling).

        [B, C, H, W]
            Spatial feature map.
            Uses global average pooling.
        """

        if features.ndim == 2:
            return features

        if features.ndim == 3:
            return features[:, 0]

        if features.ndim == 4:
            return features.mean(
                dim=(2, 3)
            )

        raise RuntimeError(
            "Unsupported TIMM feature shape: "
            f"{tuple(features.shape)}"
        )

    # ==============================================================
    # MODEL METADATA
    # ==============================================================

    def _get_embedding_dim(self) -> int | None:
        """
        Determine the output embedding dimension.
        """

        if hasattr(self.model, "num_features"):
            value = self.model.num_features

            if value is not None:
                return int(value)

        if hasattr(self.model, "embed_dim"):
            value = self.model.embed_dim

            if value is not None:
                return int(value)

        return None

    def _get_input_size(self) -> int:
        """
        Determine expected model input size.
        """

        if hasattr(self.model, "patch_embed"):

            patch_embed = self.model.patch_embed

            if hasattr(patch_embed, "img_size"):

                img_size = patch_embed.img_size

                if isinstance(img_size, tuple):
                    return int(img_size[0])

                return int(img_size)

        return 224

    # ==============================================================
    # PUBLIC MODEL INFO
    # ==============================================================

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="ViT-B/16",
            model_id=self.model_id,
            embedding_dim=self.embedding_dim,
            input_size=self.input_size,
            modality="histopathology",
            description=(
                "Generic ImageNet-pretrained "
                "Vision Transformer baseline."
            ),
        )
