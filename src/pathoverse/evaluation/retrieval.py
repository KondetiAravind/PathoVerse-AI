from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    queries: int


class RetrievalEvaluator:
    """
    Evaluate image-to-image semantic retrieval.

    Each tile has a semantic category.

    For a query tile:
      1. The query tile itself is excluded.
      2. Retrieved tiles are checked against the query category.
      3. Recall@K is computed as:

         relevant retrieved tiles / total relevant
         candidate tiles for that category
    """

    def __init__(
        self,
        labels: dict[int, str],
    ):
        self.labels = {
            int(tile_id): category
            for tile_id, category in labels.items()
        }

    def relevant_tiles(
        self,
        query_tile_id: int,
    ) -> set[int]:

        if query_tile_id not in self.labels:
            raise ValueError(
                f"No label found for query tile "
                f"{query_tile_id}"
            )

        category = self.labels[query_tile_id]

        return {
            tile_id
            for tile_id, tile_category
            in self.labels.items()
            if tile_category == category
            and tile_id != query_tile_id
        }

    def recall_at_k(
        self,
        retrieved_tile_ids: Iterable[int],
        query_tile_id: int,
        k: int,
    ) -> float:

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        relevant = self.relevant_tiles(
            query_tile_id
        )

        if not relevant:
            return 0.0

        # --------------------------------------------------
        # Remove the query tile itself.
        # --------------------------------------------------

        filtered = [
            tile_id
            for tile_id in retrieved_tile_ids
            if tile_id != query_tile_id
        ]

        retrieved = filtered[:k]

        hits = sum(
            1
            for tile_id in retrieved
            if tile_id in relevant
        )

        return min(
            hits / len(relevant),
            1.0,
        )

    def evaluate_query(
        self,
        retrieved_tile_ids: Iterable[int],
        query_tile_id: int,
    ) -> dict:

        retrieved = list(
            retrieved_tile_ids
        )

        return {
            "query_tile_id": query_tile_id,
            "category": self.labels[
                query_tile_id
            ],
            "relevant_tile_count": len(
                self.relevant_tiles(
                    query_tile_id
                )
            ),
            "recall_at_1": self.recall_at_k(
                retrieved,
                query_tile_id,
                1,
            ),
            "recall_at_5": self.recall_at_k(
                retrieved,
                query_tile_id,
                5,
            ),
            "recall_at_10": self.recall_at_k(
                retrieved,
                query_tile_id,
                10,
            ),
        }

    def evaluate(
        self,
        retrieval_results: dict[int, list[int]],
    ) -> RetrievalMetrics:

        if not retrieval_results:
            raise ValueError(
                "No retrieval results supplied."
            )

        values = [
            self.evaluate_query(
                retrieved_ids,
                query_tile_id,
            )
            for query_tile_id, retrieved_ids
            in retrieval_results.items()
        ]

        return RetrievalMetrics(
            recall_at_1=float(
                np.mean([
                    item["recall_at_1"]
                    for item in values
                ])
            ),
            recall_at_5=float(
                np.mean([
                    item["recall_at_5"]
                    for item in values
                ])
            ),
            recall_at_10=float(
                np.mean([
                    item["recall_at_10"]
                    for item in values
                ])
            ),
            queries=len(values),
        )


def build_labels_from_ground_truth(
    ground_truth: dict,
) -> dict[int, str]:
    """
    Convert ground_truth.json category assignments into:

        {
            tile_id: category
        }
    """

    labels: dict[int, str] = {}

    categories = ground_truth["categories"]

    for category, category_data in categories.items():

        for tile_id in category_data[
            "relevant_tiles"
        ]:

            tile_id = int(tile_id)

            if tile_id in labels:
                raise ValueError(
                    f"Tile {tile_id} appears in "
                    f"multiple categories."
                )

            labels[tile_id] = category

    return labels
