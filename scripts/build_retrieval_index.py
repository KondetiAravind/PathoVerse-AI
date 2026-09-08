from __future__ import annotations

import json

import numpy as np

from pathoverse.models import FAISSRetriever


EMBEDDING_PATH = (
    "data/processed/embeddings/"
    "CMU-1-Small-Region_vit-base.npy"
)

METADATA_PATH = (
    "data/metadata/embeddings/"
    "CMU-1-Small-Region_vit-base.json"
)

INDEX_PATH = (
    "data/processed/embeddings/"
    "CMU-1-Small-Region_vit-base.faiss"
)


def main():

    print("=" * 72)
    print("PATHOVERSE AI — FAISS RETRIEVAL INDEX")
    print("=" * 72)

    embeddings = np.load(
        EMBEDDING_PATH
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    print("Embeddings :", embeddings.shape)
    print("Metadata   :", len(metadata))

    retriever = FAISSRetriever()

    retriever.build(
        embeddings,
        metadata,
    )

    retriever.save(
        INDEX_PATH
    )

    print("\nIndex:")
    print(INDEX_PATH)

    query = embeddings[0]

    results = retriever.search(
        query,
        top_k=5,
    )

    print("\nTop-5 similar tiles:")

    for result in results:

        print(
            f"Rank {result.rank} | "
            f"Tile {result.tile_id} | "
            f"Score {result.score:.4f} | "
            f"Location ({result.x}, {result.y})"
        )

    print("\n✓ FAISS RETRIEVAL INDEX READY")


if __name__ == "__main__":
    main()