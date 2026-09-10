from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from pathoverse.api.dependencies import get_processed_root
from pathoverse.api.services.model_service import normalize_model_id
from pathoverse.api.services.wsi_service import wsi_service


# Physical artifact names on disk.
#
# These are intentionally different from the canonical
# PathoVerse API model IDs in some cases.
MODEL_FILES = {
    "vit-b-16": "CMU-1-Small-Region_vit-base",
    "gigapath-flash": "CMU-1-Small-Region_gigapath-flash",
    "conch": "CMU-1-Small-Region_conch",
}


class RetrievalService:
    """FAISS-backed retrieval over precomputed WSI embeddings."""

    def __init__(self) -> None:
        self.embedding_root = (
            get_processed_root() / "embeddings"
        )

    def _base_name(
        self,
        slide_id: str,
        model: str,
    ) -> str:
        """
        Resolve a canonical PathoVerse model ID to the
        physical embedding artifact basename.
        """

        canonical_model = normalize_model_id(model)

        if slide_id == "CMU-1-Small-Region":
            if canonical_model in MODEL_FILES:
                return MODEL_FILES[canonical_model]

        return f"{slide_id}_{canonical_model}"

    def _embedding_path(
        self,
        slide_id: str,
        model: str,
    ) -> Path:
        return (
            self.embedding_root
            / f"{self._base_name(slide_id, model)}.npy"
        )

    def _index_path(
        self,
        slide_id: str,
        model: str,
    ) -> Path:
        return (
            self.embedding_root
            / f"{self._base_name(slide_id, model)}.faiss"
        )

    def _load_embeddings(
        self,
        slide_id: str,
        model: str,
    ) -> np.ndarray:
        path = self._embedding_path(
            slide_id,
            model,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {path}"
            )

        embeddings = np.load(path)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings, got {embeddings.shape}"
            )

        return embeddings.astype(
            np.float32,
            copy=False,
        )

    def _load_index(
        self,
        slide_id: str,
        model: str,
    ):
        path = self._index_path(
            slide_id,
            model,
        )

        if not path.exists():
            return None

        return faiss.read_index(
            str(path)
        )

    def search(
        self,
        slide_id: str,
        tile_id: int,
        model: str,
        top_k: int,
    ) -> list[dict]:
        """
        Search for the most similar tiles to a query tile.
        """

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        embeddings = self._load_embeddings(
            slide_id,
            model,
        )

        if tile_id < 0 or tile_id >= len(embeddings):
            raise IndexError(
                f"Tile ID {tile_id} outside embedding range."
            )

        # ------------------------------------------------------
        # Query embedding
        # ------------------------------------------------------

        query = embeddings[
            tile_id : tile_id + 1
        ].copy()

        faiss.normalize_L2(query)

        # ------------------------------------------------------
        # Load existing FAISS index if available.
        # Otherwise build an in-memory cosine-similarity index.
        # ------------------------------------------------------

        index = self._load_index(
            slide_id,
            model,
        )

        if index is None:
            index = faiss.IndexFlatIP(
                embeddings.shape[1]
            )

            normalized = embeddings.copy()

            faiss.normalize_L2(
                normalized
            )

            index.add(
                normalized
            )

        search_count = min(
            top_k + 1,
            len(embeddings),
        )

        scores, indices = index.search(
            query,
            search_count,
        )

        # ------------------------------------------------------
        # Tile metadata
        # ------------------------------------------------------

        tiles = {
            int(tile["tile_id"]): tile
            for tile in wsi_service.get_tiles(
                slide_id
            )
        }

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):
            index_id = int(index_id)

            # Remove the query tile itself.
            if index_id == tile_id:
                continue

            tile = tiles.get(index_id)

            if tile is None:
                continue

            results.append(
                {
                    "tile_id": index_id,
                    "slide_id": slide_id,
                    "score": float(score),
                    "x": int(tile["x"]),
                    "y": int(tile["y"]),
                    "width": int(tile["width"]),
                    "height": int(tile["height"]),
                }
            )

            if len(results) >= top_k:
                break

        return results


retrieval_service = RetrievalService()