from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pathoverse.evaluation.retrieval import (
    RetrievalEvaluator,
    build_labels_from_ground_truth,
)
from pathoverse.models.retrieval import FAISSRetriever


GROUND_TRUTH_PATH = Path(
    "results/retrieval/ground_truth.json"
)

EMBEDDING_DIR = Path(
    "data/processed/embeddings"
)

OUTPUT_DIR = Path(
    "results/retrieval"
)


MODELS = {
    "ViT-B/16": (
        "CMU-1-Small-Region_vit-base.npy"
    ),
    "GigaPath-Flash": (
        "CMU-1-Small-Region_gigapath-flash.npy"
    ),
    "CONCH": (
        "CMU-1-Small-Region_conch.npy"
    ),
}


def load_ground_truth():
    with open(
        GROUND_TRUTH_PATH,
        "r",
    ) as f:
        return json.load(f)


def load_tile_metadata():
    path = Path(
        "data/metadata/tiles/extracted_tiles.json"
    )

    with open(path, "r") as f:
        data = json.load(f)

    return data["tiles"]


def build_retriever(
    embeddings: np.ndarray,
    tiles: list[dict],
):
    retriever = FAISSRetriever()

    retriever.build(
        embeddings,
        tiles,
    )

    return retriever


def evaluate_model(
    model_name: str,
    embedding_file: str,
    labels: dict[int, str],
    tiles: list[dict],
):
    print()
    print("=" * 72)
    print(f"EVALUATING: {model_name}")
    print("=" * 72)

    embedding_path = (
        EMBEDDING_DIR / embedding_file
    )

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Missing embeddings: "
            f"{embedding_path}"
        )

    embeddings = np.load(
        embedding_path
    ).astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, "
            f"got {embeddings.shape}"
        )

    if len(embeddings) != len(tiles):
        raise ValueError(
            f"Embedding/tile mismatch: "
            f"{len(embeddings)} vs "
            f"{len(tiles)}"
        )

    print(
        "Embeddings:",
        embeddings.shape,
    )

    retriever = build_retriever(
        embeddings,
        tiles,
    )

    evaluator = RetrievalEvaluator(
        labels
    )

    # ------------------------------------------------------
    # Evaluate every labeled tile.
    # ------------------------------------------------------

    per_query = []

    for query_tile_id in sorted(labels):

        # Retrieve K+1 because the query tile itself
        # normally appears at rank 1.
        results = retriever.search(
            embeddings[query_tile_id],
            top_k=11,
        )

        retrieved_ids = [
            result.tile_id
            for result in results
        ]

        result = evaluator.evaluate_query(
            retrieved_ids,
            query_tile_id,
        )

        per_query.append(result)

    # ------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------

    recall_at_1 = float(
        np.mean([
            result["recall_at_1"]
            for result in per_query
        ])
    )

    recall_at_5 = float(
        np.mean([
            result["recall_at_5"]
            for result in per_query
        ])
    )

    recall_at_10 = float(
        np.mean([
            result["recall_at_10"]
            for result in per_query
        ])
    )

    # ------------------------------------------------------
    # Per-category metrics
    # ------------------------------------------------------

    categories = sorted(
        set(labels.values())
    )

    per_category = {}

    for category in categories:

        category_results = [
            result
            for result in per_query
            if result["category"] == category
        ]

        per_category[category] = {
            "queries": len(
                category_results
            ),
            "recall_at_1": float(
                np.mean([
                    x["recall_at_1"]
                    for x in category_results
                ])
            ),
            "recall_at_5": float(
                np.mean([
                    x["recall_at_5"]
                    for x in category_results
                ])
            ),
            "recall_at_10": float(
                np.mean([
                    x["recall_at_10"]
                    for x in category_results
                ])
            ),
        }

    result = {
        "schema_version": "1.0",

        "model": {
            "name": model_name,
            "embedding_file": embedding_file,
            "embedding_dimension": int(
                embeddings.shape[1]
            ),
        },

        "dataset": {
            "slide_id": (
                "CMU-1-Small-Region"
            ),
            "tiles": len(tiles),
            "queries": len(per_query),
            "ground_truth": str(
                GROUND_TRUTH_PATH
            ),
        },

        "protocol": {
            "type": (
                "leave_one_out_image_to_image"
            ),
            "retrieval_k": 11,
            "query_tile_excluded": True,
            "evaluated_k": [
                1,
                5,
                10,
            ],
        },

        "overall": {
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
        },

        "per_category": per_category,

        "per_query": per_query,
    }

    safe_name = (
        model_name
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

    output_path = (
        OUTPUT_DIR
        / f"{safe_name}_retrieval_evaluation.json"
    )

    with open(
        output_path,
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print(
        f"Recall@1  : {recall_at_1:.4f}"
    )
    print(
        f"Recall@5  : {recall_at_5:.4f}"
    )
    print(
        f"Recall@10 : {recall_at_10:.4f}"
    )

    print()
    print("Per-category:")

    for category, metrics in (
        per_category.items()
    ):
        print(
            f"  {category:<30}"
            f"R@1={metrics['recall_at_1']:.4f} "
            f"R@5={metrics['recall_at_5']:.4f} "
            f"R@10={metrics['recall_at_10']:.4f}"
        )

    print()
    print("Saved:")
    print(output_path)

    return result


def main():

    print("=" * 72)
    print(
        "PATHOVERSE — IMAGE RETRIEVAL EVALUATION"
    )
    print("=" * 72)

    ground_truth = load_ground_truth()

    labels = build_labels_from_ground_truth(
        ground_truth
    )

    tiles = load_tile_metadata()

    if len(labels) != len(tiles):
        raise ValueError(
            f"Ground-truth labels: {len(labels)}, "
            f"tiles: {len(tiles)}"
        )

    print(
        "Ground-truth tiles:",
        len(labels),
    )

    results = {}

    for model_name, embedding_file in (
        MODELS.items()
    ):

        results[model_name] = evaluate_model(
            model_name,
            embedding_file,
            labels,
            tiles,
        )

    # ------------------------------------------------------
    # Summary leaderboard
    # ------------------------------------------------------

    leaderboard = []

    for model_name, result in (
        results.items()
    ):

        leaderboard.append(
            {
                "model": model_name,
                "embedding_dimension": result[
                    "model"
                ]["embedding_dimension"],
                "recall_at_1": result[
                    "overall"
                ]["recall_at_1"],
                "recall_at_5": result[
                    "overall"
                ]["recall_at_5"],
                "recall_at_10": result[
                    "overall"
                ]["recall_at_10"],
            }
        )

    leaderboard.sort(
        key=lambda x: x["recall_at_10"],
        reverse=True,
    )

    leaderboard_output = {
        "schema_version": "1.0",
        "evaluation": "image_to_image",
        "ground_truth_status": ground_truth[
            "dataset"
        ]["annotation_status"],
        "models": leaderboard,
    }

    leaderboard_path = (
        OUTPUT_DIR
        / "retrieval_leaderboard.json"
    )

    with open(
        leaderboard_path,
        "w",
    ) as f:
        json.dump(
            leaderboard_output,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "PATHOVERSE — RETRIEVAL LEADERBOARD"
    )
    print("=" * 72)

    print(
        f"{'Rank':<6}"
        f"{'Model':<22}"
        f"{'Dim':>8}"
        f"{'R@1':>10}"
        f"{'R@5':>10}"
        f"{'R@10':>10}"
    )

    print("-" * 72)

    for rank, row in enumerate(
        leaderboard,
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"{row['model']:<22}"
            f"{row['embedding_dimension']:>8}"
            f"{row['recall_at_1']:>10.4f}"
            f"{row['recall_at_5']:>10.4f}"
            f"{row['recall_at_10']:>10.4f}"
        )

    print()
    print(
        "Saved:",
        leaderboard_path,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
