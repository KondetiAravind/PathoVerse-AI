from __future__ import annotations

from typing import Dict, Type

from .base import BasePathologyDataset


class DatasetRegistry:
    """
    Central registry for PathoVerse datasets.
    """

    def __init__(self):
        self._datasets: Dict[str, Type[BasePathologyDataset]] = {}

    def register(self, name: str):
        """
        Decorator used to register a dataset adapter.
        """

        def decorator(cls: Type[BasePathologyDataset]):
            if name in self._datasets:
                raise ValueError(
                    f"Dataset '{name}' is already registered."
                )

            self._datasets[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type[BasePathologyDataset]:
        if name not in self._datasets:
            available = ", ".join(sorted(self._datasets))
            raise KeyError(
                f"Unknown dataset '{name}'. "
                f"Available datasets: {available}"
            )

        return self._datasets[name]

    def list(self) -> list[str]:
        return sorted(self._datasets.keys())


DATASET_REGISTRY = DatasetRegistry()


def register_dataset(name: str):
    return DATASET_REGISTRY.register(name)