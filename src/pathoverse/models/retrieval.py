from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class RetrievalResult:
    rank: int
    tile_id: int
    slide_id: str
    score: float
    x: int
    y: int
    image_path: str


class FAISSRetriever:
    """
    FAISS-based similarity search over pathology tile embeddings.

    Embeddings are L2-normalized before indexing, so inner product
    corresponds to cosine similarity.
    """

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

    def build(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must have shape [N, D]."
            )

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Embedding count and metadata count differ."
            )

        vectors = embeddings.copy()

        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(vectors)

        self.metadata = list(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if self.index is None:
            raise RuntimeError(
                "FAISS index has not been built."
            )

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        faiss.normalize_L2(query)

        scores, indices = self.index.search(
            query,
            min(top_k, self.index.ntotal),
        )

        results = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):

            if idx < 0:
                continue

            item = self.metadata[idx]

            results.append(
                RetrievalResult(
                    rank=rank,
                    tile_id=int(
                        item["tile_id"]
                    ),
                    slide_id=item["slide_id"],
                    score=float(score),
                    x=int(item.get("x", 0)),
                    y=int(item.get("y", 0)),
                    image_path=item.get(
                        "image_path",
                        "",
                    ),
                )
            )

        return results

    def save(
        self,
        index_path: str | Path,
    ) -> None:

        if self.index is None:
            raise RuntimeError(
                "Cannot save an empty index."
            )

        index_path = Path(index_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

    def load(
        self,
        index_path: str | Path,
        metadata: list[dict],
    ) -> None:

        self.index = faiss.read_index(
            str(index_path)
        )

        self.metadata = list(metadata)