from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image

from pathoverse.models.base import BaseModelAdapter, ModelInfo


class ConchAdapter(BaseModelAdapter):
    """
    PathoVerse adapter for MahmoodLab CONCH.

    CONCH is a pathology vision-language foundation model.
    This adapter exposes:

        image -> 512-D contrastive embedding
        text  -> 512-D contrastive embedding

    The contrastive embedding space is shared between images
    and text, enabling image-text and text-image retrieval.
    """

    def __init__(
        self,
        model_id: str = "conch_ViT-B-16",
        checkpoint_id: str = "hf_hub:MahmoodLab/conch",
        device: str = "cuda",
        batch_size: int = 4,
        hf_auth_token: str | None = None,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
        )

        self.checkpoint_id = checkpoint_id
        self.hf_auth_token = hf_auth_token

        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def load(self) -> None:
        """
        Load the pretrained CONCH model from Hugging Face.
        """

        from conch.open_clip_custom import (
            create_model_from_pretrained,
        )

        self.model, self.preprocess = create_model_from_pretrained(
            self.model_id,
            self.checkpoint_id,
            device=str(self.device),
            hf_auth_token=self.hf_auth_token,
        )

        self.model.eval()

        from conch.open_clip_custom import get_tokenizer

        self.tokenizer = get_tokenizer()

    def preprocess_images(
        self,
        images: Sequence[Image.Image],
    ) -> torch.Tensor:
        """
        Convert PIL images into the tensor expected by CONCH.
        """

        self.ensure_loaded()

        processed = []

        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            image = image.convert("RGB")
            processed.append(self.preprocess(image))

        return torch.stack(processed)

    def preprocess(
        self,
        images: Sequence[Image.Image],
    ) -> torch.Tensor:
        """
        Compatibility wrapper for BaseModelAdapter.
        """

        return self.preprocess_images(images)

    @torch.inference_mode()
    def encode(
        self,
        images: Sequence[Image.Image] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate 512-D CONCH contrastive image embeddings.
        """

        self.ensure_loaded()

        if isinstance(images, torch.Tensor):
            batch = images.to(self.device)
        else:
            batch = self.preprocess_images(images).to(self.device)

        embeddings = self.model.encode_image(
            batch,
            normalize=True,
            proj_contrast=True,
        )

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D image embeddings, got "
                f"shape={tuple(embeddings.shape)}"
            )

        if embeddings.shape[1] != 512:
            raise RuntimeError(
                f"Expected CONCH 512-D contrastive embeddings, "
                f"got {embeddings.shape[1]}"
            )

        return embeddings.float()

    @torch.inference_mode()
    def encode_text(
        self,
        texts: Sequence[str],
    ) -> torch.Tensor:
        """
        Generate normalized 512-D CONCH text embeddings.

        Uses the CONCH tokenizer directly through its modern
        Hugging Face tokenizer interface. This avoids the
        batch_encode_plus compatibility issue in the original
        CONCH helper with newer Transformers versions.
        """

        self.ensure_loaded()

        if not texts:
            raise ValueError("texts must not be empty")

        encoded = self.tokenizer(
            list(texts),
            max_length=127,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        tokens = encoded["input_ids"]

        # CONCH expects a sequence length of 128.
        # Its original tokenizer helper pads one additional
        # token after producing the 127-token Hugging Face output.
        tokens = torch.nn.functional.pad(
            tokens,
            (0, 1),
            value=self.tokenizer.pad_token_id,
        )

        tokens = tokens.to(self.device)

        embeddings = self.model.encode_text(
            tokens,
            normalize=True,
        )

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D text embeddings, got "
                f"shape={tuple(embeddings.shape)}"
            )

        if embeddings.shape[1] != 512:
            raise RuntimeError(
                f"Expected CONCH 512-D text embeddings, "
                f"got {embeddings.shape[1]}"
            )

        return embeddings.float()

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="CONCH",
            model_id=self.model_id,
            embedding_dim=512,
            input_size=448,
            modality="vision-language",
            description=(
                "CONCH pathology vision-language foundation model "
                "with shared 512-D image-text embeddings"
            ),
        )
