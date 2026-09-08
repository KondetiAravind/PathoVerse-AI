from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DatasetInfo:
    name: str
    description: str
    num_classes: int
    class_names: list[str]
    image_shape: tuple[int, ...]
    split_names: list[str]
    root_dir: Path


class BasePathologyDataset(ABC):
    """
    Standard interface for every PathoVerse dataset.

    Each dataset adapter must implement:
    - download
    - validate
    - statistics
    - get_split
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique dataset identifier."""
        raise NotImplementedError

    @abstractmethod
    def download(self, **kwargs: Any) -> None:
        """Download or prepare the dataset."""
        raise NotImplementedError

    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """Validate dataset files and structure."""
        raise NotImplementedError

    @abstractmethod
    def statistics(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        raise NotImplementedError

    @abstractmethod
    def get_split(self, split: str):
        """Return a dataset split."""
        raise NotImplementedError

    @abstractmethod
    def info(self) -> DatasetInfo:
        """Return standardized dataset metadata."""
        raise NotImplementedError