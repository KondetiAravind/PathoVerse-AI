from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelDefinition:
    key: str
    name: str
    model_id: str
    embedding_dimension: int
    input_size: int
    modality: str
    description: str


MODEL_DEFINITIONS = {
    "vit-b-16": ModelDefinition(
        key="vit-b-16",
        name="ViT-B/16",
        model_id="google/vit-base-patch16-224-in21k",
        embedding_dimension=768,
        input_size=224,
        modality="vision",
        description=(
            "Vision Transformer baseline foundation encoder."
        ),
    ),
    "gigapath-flash": ModelDefinition(
        key="gigapath-flash",
        name="GigaPath-Flash",
        model_id="prov-gigapath/gigapath-flash",
        embedding_dimension=384,
        input_size=224,
        modality="histopathology",
        description=(
            "Efficient pathology foundation encoder."
        ),
    ),
    "conch": ModelDefinition(
        key="conch",
        name="CONCH",
        model_id="MahmoodLab/CONCH",
        embedding_dimension=512,
        input_size=448,
        modality="vision-language",
        description=(
            "Pathology vision-language foundation model."
        ),
    ),
}


# Backward-compatible aliases.
MODEL_ALIASES = {
    "vit-base": "vit-b-16",
    "vit_b_16": "vit-b-16",
    "vit-b/16": "vit-b-16",
    "ViT-B/16": "vit-b-16",
    "gigapath_flash": "gigapath-flash",
    "GigaPath-Flash": "gigapath-flash",
    "CONCH": "conch",
}


def normalize_model_id(model: str) -> str:
    """Return the canonical PathoVerse model ID."""

    value = model.strip()

    if value in MODEL_DEFINITIONS:
        return value

    return MODEL_ALIASES.get(value, value)


class ModelService:
    """Model catalog and lazy-loading service."""

    def __init__(self) -> None:
        self._loaded_models: dict[str, Any] = {}

    def list_models(self) -> list[ModelDefinition]:
        return list(MODEL_DEFINITIONS.values())

    def get_definition(self, key: str) -> ModelDefinition:
        canonical = normalize_model_id(key)

        if canonical not in MODEL_DEFINITIONS:
            raise KeyError(
                f"Unknown model '{key}'."
            )

        return MODEL_DEFINITIONS[canonical]

    def is_loaded(self, key: str) -> bool:
        canonical = normalize_model_id(key)
        return canonical in self._loaded_models

    def loaded_models(self) -> list[str]:
        return list(self._loaded_models.keys())

    def status(self, key: str) -> dict[str, Any]:
        definition = self.get_definition(key)

        return {
            "key": definition.key,
            "name": definition.name,
            "model_id": definition.model_id,
            "embedding_dimension": (
                definition.embedding_dimension
            ),
            "input_size": definition.input_size,
            "modality": definition.modality,
            "loaded": self.is_loaded(
                definition.key
            ),
        }

    def unload(self, key: str) -> None:
        canonical = normalize_model_id(key)
        self._loaded_models.pop(
            canonical,
            None,
        )


model_service = ModelService()