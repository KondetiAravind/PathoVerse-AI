from __future__ import annotations

import json
from pathlib import Path


RESULTS_DIR = Path(
    "results/classification"
)


MODELS = [
    "vit_b_16",
    "gigapath_flash",
    "conch",
]


def main():

    rows = []

    for model in MODELS:

        path = (
            RESULTS_DIR
            / f"{model}_classification.json"
        )

        if not path.exists():

            print(
                f"Skipping missing result: {path}"
            )

            continue

        with open(
            path,
            "r",
        ) as f:

            result = json.load(f)

        test = result["test"]

        model_info = result["model"]

        rows.append(
            {
                "model": model_info["name"],
                "model_id": model,
                "embedding_dimension": (
                    model_info[
                        "embedding_dimension"
                    ]
                ),
                "accuracy": test["accuracy"],
                "auroc": test["auroc"],
                "f1": test["f1"],
                "precision": test["precision"],
                "recall": test["recall"],
                "sensitivity": test["sensitivity"],
                "specificity": test["specificity"],
            }
        )

    rows.sort(
        key=lambda x: x["auroc"],
        reverse=True,
    )

    for rank, row in enumerate(
        rows,
        start=1,
    ):

        row["rank"] = rank

    output = {
        "schema_version": "1.0",
        "task": "patchcamelyon_binary_classification",
        "ranking_metric": "test_auroc",
        "models": rows,
    }

    output_path = (
        RESULTS_DIR
        / "classification_leaderboard.json"
    )

    with open(
        output_path,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 88)
    print(
        "PATHOVERSE — CLASSIFICATION LEADERBOARD"
    )
    print("=" * 88)

    print(
        f"{'Rank':<6}"
        f"{'Model':<22}"
        f"{'Dim':<8}"
        f"{'Accuracy':<12}"
        f"{'AUROC':<12}"
        f"{'F1':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
    )

    print("-" * 88)

    for row in rows:

        print(
            f"{row['rank']:<6}"
            f"{row['model']:<22}"
            f"{row['embedding_dimension']:<8}"
            f"{row['accuracy']:<12.4f}"
            f"{row['auroc']:<12.4f}"
            f"{row['f1']:<12.4f}"
            f"{row['precision']:<12.4f}"
            f"{row['recall']:<12.4f}"
        )

    print()
    print(
        "Saved:",
        output_path,
    )

    print("=" * 88)


if __name__ == "__main__":
    main()