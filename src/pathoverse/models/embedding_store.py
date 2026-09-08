from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class EmbeddingStore:
    """
    Persistent storage for model embeddings and metadata.
    """

    def __init__(
        self,
        embedding_dir: str = "data/processed/embeddings",
        metadata_dir: str = "data/metadata/embeddings",
    ) -> None:
        self.embedding_dir = Path(embedding_dir)
        self.metadata_dir = Path(metadata_dir)

        self.embedding_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        embeddings: torch.Tensor | np.ndarray,
        metadata: list[dict],
        name: str,
    ) -> tuple[Path, Path]:

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        embedding_path = (
            self.embedding_dir / f"{name}.npy"
        )

        metadata_path = (
            self.metadata_dir / f"{name}.json"
        )

        np.save(
            embedding_path,
            embeddings,
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

        return embedding_path, metadata_path

    def load(
        self,
        name: str,
    ) -> tuple[np.ndarray, list[dict]]:

        embedding_path = (
            self.embedding_dir / f"{name}.npy"
        )

        metadata_path = (
            self.metadata_dir / f"{name}.json"
        )

        embeddings = np.load(
            embedding_path,
        )

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        return embeddings, metadata