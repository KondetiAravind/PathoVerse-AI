from pathoverse.evaluation.retrieval import (
    RetrievalEvaluator,
    build_labels_from_ground_truth,
)

def test_query_tile_is_excluded():

    labels = {
        0: "epithelial",
        1: "epithelial",
        2: "epithelial",
        3: "stroma",
    }

    evaluator = RetrievalEvaluator(labels)

    retrieved = [
        0,
        1,
        2,
        3,
    ]

    score = evaluator.recall_at_k(
        retrieved,
        query_tile_id=0,
        k=1,
    )

    # Tile 0 must be excluded.
    # After exclusion:
    # [1, 2, 3]
    #
    # Top-1 = [1]
    # Relevant tiles = {1, 2}
    # Recall@1 = 1 / 2 = 0.5

    assert score == 0.5

def test_recall_at_five():

    labels = {
        0: "epithelial",
        1: "epithelial",
        2: "epithelial",
        3: "epithelial",
        4: "stroma",
    }

    evaluator = RetrievalEvaluator(labels)

    retrieved = [
        0,
        1,
        4,
        2,
        3,
    ]

    score = evaluator.recall_at_k(
        retrieved,
        query_tile_id=0,
        k=5,
    )

    assert score == 1.0


def test_category_relevant_tiles_exclude_query():

    labels = {
        0: "epithelial",
        1: "epithelial",
        2: "epithelial",
        3: "stroma",
    }

    evaluator = RetrievalEvaluator(labels)

    relevant = evaluator.relevant_tiles(0)

    assert relevant == {1, 2}


def test_ground_truth_conversion():

    ground_truth = {
        "categories": {
            "epithelial_rich": {
                "relevant_tiles": [0, 1]
            },
            "stroma_collagen_rich": {
                "relevant_tiles": [2]
            }
        }
    }

    labels = build_labels_from_ground_truth(
        ground_truth
    )

    assert labels == {
        0: "epithelial_rich",
        1: "epithelial_rich",
        2: "stroma_collagen_rich",
    }


def test_ground_truth_duplicate_detection():

    ground_truth = {
        "categories": {
            "epithelial_rich": {
                "relevant_tiles": [0, 1]
            },
            "stroma_collagen_rich": {
                "relevant_tiles": [1, 2]
            }
        }
    }

    try:
        build_labels_from_ground_truth(
            ground_truth
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected duplicate tile detection."
    )

def test_query_is_removed_before_top_k():

    labels = {
        0: "epithelial",
        1: "epithelial",
        2: "epithelial",
        3: "epithelial",
        4: "stroma",
    }

    evaluator = RetrievalEvaluator(labels)

    retrieved = [
        0,
        1,
        2,
        3,
    ]

    # Relevant tiles excluding query:
    # {1, 2, 3}
    #
    # After removing query:
    # [1, 2, 3]
    #
    # Top-1:
    # [1]
    #
    # Recall@1 = 1 / 3

    score = evaluator.recall_at_k(
        retrieved,
        query_tile_id=0,
        k=1,
    )

    assert abs(
        score - (1 / 3)
    ) < 1e-8