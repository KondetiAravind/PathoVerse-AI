from __future__ import annotations

from typing import Callable

from pathoverse.models.adapters.hf_vision import HFVisionAdapter


class ModelRegistry:
    """
    Central registry for PathoVerse foundation model adapters.
    """

    def __init__(self) -> None:
        self._models: dict[str, Callable] = {}

    def register(self, name: str, factory: Callable) -> None:
        if name in self._models:
            raise ValueError(
                f"Model '{name}' is already registered."
            )

        self._models[name] = factory

    def create(self, name: str, **kwargs):
        if name not in self._models:
            available = ", ".join(sorted(self._models))
            raise KeyError(
                f"Unknown model '{name}'. "
                f"Available models: {available}"
            )

        return self._models[name](**kwargs)

    def list_models(self) -> list[str]:
        return sorted(self._models)


registry = ModelRegistry()


registry.register(
    "vit-base",
    lambda **kwargs: HFVisionAdapter(
        model_id="google/vit-base-patch16-224-in21k",
        **kwargs,
    ),
)