from __future__ import annotations

import numpy as np
import pytest

from pathoverse.models.retrieval import (
    FAISSRetriever,
)


def sample_metadata(count: int = 5) -> list[dict]:
    return [
        {
            "tile_id": i,
            "slide_id": "test-slide",
            "x": i * 224,
            "y": i * 224,
            "image_path": f"tile_{i}.png",
        }
        for i in range(count)
    ]


def test_build_and_search():

    embeddings = np.eye(5, dtype=np.float32)

    metadata = sample_metadata(5)

    retriever = FAISSRetriever()

    retriever.build(
        embeddings,
        metadata,
    )

    results = retriever.search(
        embeddings[0],
        top_k=3,
    )

    assert len(results) == 3

    assert results[0].rank == 1
    assert results[0].tile_id == 0

    assert results[0].score == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_search_returns_descending_scores():

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    metadata = sample_metadata(3)

    retriever = FAISSRetriever()

    retriever.build(
        embeddings,
        metadata,
    )

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = retriever.search(
        query,
        top_k=3,
    )

    scores = [
        result.score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_search_respects_top_k():

    embeddings = np.eye(
        10,
        dtype=np.float32,
    )

    metadata = sample_metadata(10)

    retriever = FAISSRetriever()

    retriever.build(
        embeddings,
        metadata,
    )

    results = retriever.search(
        embeddings[0],
        top_k=3,
    )

    assert len(results) == 3


def test_search_before_build_fails():

    retriever = FAISSRetriever()

    with pytest.raises(RuntimeError):
        retriever.search(
            np.array([1.0, 0.0]),
            top_k=5,
        )


def test_build_dimension_and_metadata_validation():

    retriever = FAISSRetriever()

    embeddings = np.eye(
        5,
        dtype=np.float32,
    )

    metadata = sample_metadata(4)

    with pytest.raises(ValueError):
        retriever.build(
            embeddings,
            metadata,
        )


def test_result_contains_tile_metadata():

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    metadata = [
        {
            "tile_id": 42,
            "slide_id": "slide-A",
            "x": 896,
            "y": 448,
            "image_path": "tile.png",
        },
        {
            "tile_id": 99,
            "slide_id": "slide-A",
            "x": 1120,
            "y": 672,
            "image_path": "tile2.png",
        },
    ]

    retriever = FAISSRetriever()

    retriever.build(
        embeddings,
        metadata,
    )

    results = retriever.search(
        embeddings[0],
        top_k=1,
    )

    result = results[0]

    assert result.tile_id == 42
    assert result.slide_id == "slide-A"
    assert result.x == 896
    assert result.y == 448
    assert result.image_path == "tile.png"
