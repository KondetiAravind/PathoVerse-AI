from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingDataset:

    def __init__(
        self,
        embedding_path: str | Path,
        label_path: str | Path,
    ):
        self.embedding_path = Path(
            embedding_path
        )

        self.label_path = Path(
            label_path
        )

        if not self.embedding_path.exists():
            raise FileNotFoundError(
                self.embedding_path
            )

        if not self.label_path.exists():
            raise FileNotFoundError(
                self.label_path
            )

        self.embeddings = np.load(
            self.embedding_path
        )

        self.labels = np.load(
            self.label_path
        )

        if len(self.embeddings) != len(
            self.labels
        ):
            raise ValueError(
                "Embedding/label mismatch."
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.embeddings[index],
            self.labels[index],
        )