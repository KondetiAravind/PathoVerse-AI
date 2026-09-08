from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


EMBEDDINGS_PATH = Path(
    "data/processed/embeddings/"
    "CMU-1-Small-Region_conch.npy"
)

METADATA_PATH = Path(
    "data/metadata/embeddings/"
    "CMU-1-Small-Region_conch.json"
)

INDEX_PATH = Path(
    "data/processed/embeddings/"
    "CMU-1-Small-Region_conch.faiss"
)


def main() -> None:
    print("=" * 70)
    print("PATHOVERSE — CONCH FAISS INDEX")
    print("=" * 70)

    embeddings = np.load(
        EMBEDDINGS_PATH
    ).astype(np.float32)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    tiles = metadata["tiles"]

    print("Embeddings :", embeddings.shape)
    print("Tiles      :", len(tiles))

    if embeddings.ndim != 2:
        raise RuntimeError(
            "Expected 2-D embedding matrix."
        )

    if embeddings.shape[0] != len(tiles):
        raise RuntimeError(
            "Embedding count does not match metadata."
        )

    dimension = embeddings.shape[1]

    # CONCH embeddings are normalized.
    # Inner product therefore equals cosine similarity.
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    print()
    print("FAISS index")
    print("-" * 70)
    print("Dimension  :", dimension)
    print("Vectors    :", index.ntotal)
    print("Metric     : Inner Product")
    print("Meaning    : Cosine similarity")
    print()
    print("Saved:")
    print(" ", INDEX_PATH)

    print("=" * 70)
    print("✓ CONCH FAISS INDEX CREATED")
    print("=" * 70)


if __name__ == "__main__":
    main()
