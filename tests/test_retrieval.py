import numpy as np

from pathoverse.models import FAISSRetriever


def test_faiss_retrieval():

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    metadata = [
        {
            "tile_id": 0,
            "slide_id": "slide_a",
            "x": 100,
            "y": 200,
        },
        {
            "tile_id": 1,
            "slide_id": "slide_a",
            "x": 200,
            "y": 200,
        },
        {
            "tile_id": 2,
            "slide_id": "slide_a",
            "x": 300,
            "y": 200,
        },
    ]

    retriever = FAISSRetriever()

    retriever.build(
        embeddings,
        metadata,
    )

    results = retriever.search(
        embeddings[0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].tile_id == 0
    assert results[0].score > 0.99
    assert results[0].x == 100
    assert results[0].y == 200