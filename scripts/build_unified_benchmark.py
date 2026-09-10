from __future__ import annotations

import csv
import json
from pathlib import Path

from pathoverse.benchmark.collector import (
    BenchmarkCollector,
)

from pathoverse.benchmark.leaderboard import (
    build_classification_ranking,
    build_efficiency_ranking,
    build_model_summary,
    build_retrieval_ranking,
)


RESULTS_ROOT = Path(
    "results"
)

OUTPUT_DIR = Path(
    "results/unified"
)


EFFICIENCY_FILES = [
    RESULTS_ROOT
    / "benchmarks"
    / "vit_standard_benchmark.json",

    RESULTS_ROOT
    / "benchmarks"
    / "gigapath_standard_benchmark.json",

    RESULTS_ROOT
    / "benchmarks"
    / "conch_standard_benchmark.json",
]


RETRIEVAL_FILES = [
    RESULTS_ROOT
    / "retrieval"
    / "vit_b_16_retrieval_evaluation.json",

    RESULTS_ROOT
    / "retrieval"
    / "gigapath_flash_retrieval_evaluation.json",

    RESULTS_ROOT
    / "retrieval"
    / "conch_retrieval_evaluation.json",
]


CLASSIFICATION_FILES = [
    RESULTS_ROOT
    / "classification"
    / "vit_b_16_classification.json",

    RESULTS_ROOT
    / "classification"
    / "gigapath_flash_classification.json",

    RESULTS_ROOT
    / "classification"
    / "conch_classification.json",
]


MIL_FILES = [
    RESULTS_ROOT
    / "mil"
    / "CMU-1-Small-Region_gigapath_flash_mil.json",
]


def add_if_exists(
    collector: BenchmarkCollector,
    path: Path,
    function,
) -> None:

    if path.exists():

        print(
            f"✓ Loading {path}"
        )

        function(
            path
        )

    else:

        print(
            f"⚠ Missing {path}"
        )


def flatten_for_csv(
    records: list[dict],
) -> list[dict]:

    rows = []

    for record in records:

        row = {}

        for key, value in record.items():

            if key == "metadata":

                if isinstance(
                    value,
                    dict,
                ):

                    row[
                        "metadata_source"
                    ] = value.get(
                        "source_file"
                    )

                continue

            row[key] = value

        rows.append(
            row
        )

    return rows


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 88)
    print(
        "PATHOVERSE — UNIFIED BENCHMARK BUILDER"
    )
    print("=" * 88)

    collector = BenchmarkCollector(
        RESULTS_ROOT
    )

    print()
    print(
        "FOUNDATION MODEL EFFICIENCY"
    )
    print("-" * 88)

    for path in EFFICIENCY_FILES:

        add_if_exists(
            collector,
            path,
            collector.add_efficiency_result,
        )

    print()
    print(
        "IMAGE RETRIEVAL"
    )
    print("-" * 88)

    for path in RETRIEVAL_FILES:

        add_if_exists(
            collector,
            path,
            collector.add_retrieval_result,
        )

    print()
    print(
        "PCAM CLASSIFICATION"
    )
    print("-" * 88)

    for path in CLASSIFICATION_FILES:

        add_if_exists(
            collector,
            path,
            collector.add_classification_result,
        )

    print()
    print(
        "WSI MIL PROTOTYPE"
    )
    print("-" * 88)

    for path in MIL_FILES:

        add_if_exists(
            collector,
            path,
            collector.add_mil_result,
        )

    print()
    print(
        "VALIDATING BENCHMARK COLLECTION..."
    )

    # Strict validation for the actual production
    # PathoVerse benchmark.
    collector.validate(
        require_production_models=True
    )

    records = collector.to_list()

    print(
        f"✓ Validated {len(records)} benchmark records"
    )

    classification_ranking = (
        build_classification_ranking(
            records
        )
    )

    efficiency_ranking = (
        build_efficiency_ranking(
            records
        )
    )

    retrieval_ranking = (
        build_retrieval_ranking(
            records
        )
    )

    model_summary = (
        build_model_summary(
            records
        )
    )

    unified = {
        "schema_version": "1.0",

        "project": "PathoVerse AI",

        "description": (
            "Unified benchmark across "
            "pathology foundation models, "
            "image retrieval, classification, "
            "and WSI MIL prototype."
        ),

        "model_id_convention": {
            "canonical": True,

            "description": (
                "model_id values use canonical "
                "PathoVerse model identifiers."
            ),

            "models": {
                "vit-b-16": "ViT-B/16",
                "gigapath-flash": "GigaPath-Flash",
                "conch": "CONCH",
            },
        },

        "records": records,

        "rankings": {
            "classification": (
                classification_ranking
            ),

            "efficiency": (
                efficiency_ranking
            ),

            "retrieval": (
                retrieval_ranking
            ),
        },

        "model_summary": model_summary,
    }

    json_path = (
        OUTPUT_DIR
        / "foundation_model_benchmark.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            unified,
            f,
            indent=2,
        )

    csv_rows = flatten_for_csv(
        records
    )

    csv_path = (
        OUTPUT_DIR
        / "foundation_model_benchmark.csv"
    )

    fieldnames = [
        "model",
        "model_id",
        "task",
        "embedding_dimension",
        "dataset",
        "split",
        "samples",
        "accuracy",
        "auroc",
        "f1",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "latency_ms",
        "throughput",
        "gpu_memory_mb",
        "inference_time_sec",
        "status",
        "metadata_source",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )

    print()
    print("=" * 100)
    print(
        "PATHOVERSE — CLASSIFICATION LEADERBOARD"
    )
    print("=" * 100)

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

    print("-" * 100)

    for row in classification_ranking:

        print(
            f"{row['classification_rank']:<6}"
            f"{row['model']:<22}"
            f"{row['embedding_dimension']:<8}"
            f"{row['accuracy']:<12.4f}"
            f"{row['auroc']:<12.4f}"
            f"{row['f1']:<12.4f}"
            f"{row['precision']:<12.4f}"
            f"{row['recall']:<12.4f}"
        )

    print()
    print("=" * 100)
    print(
        "PATHOVERSE — EFFICIENCY LEADERBOARD"
    )
    print("=" * 100)

    print(
        f"{'Rank':<6}"
        f"{'Model':<22}"
        f"{'Dim':<8}"
        f"{'Latency ms':<14}"
        f"{'Throughput':<16}"
        f"{'GPU MB':<14}"
    )

    print("-" * 100)

    for row in efficiency_ranking:

        print(
            f"{row['efficiency_rank']:<6}"
            f"{row['model']:<22}"
            f"{row['embedding_dimension']:<8}"
            f"{row['latency_ms']:<14.3f}"
            f"{row['throughput']:<16.2f}"
            f"{row['gpu_memory_mb']:<14.2f}"
        )

    print()
    print("=" * 100)
    print(
        "PATHOVERSE — RETRIEVAL LEADERBOARD"
    )
    print("=" * 100)

    print(
        f"{'Rank':<6}"
        f"{'Model':<22}"
        f"{'R@1':<12}"
        f"{'R@5':<12}"
        f"{'R@10':<12}"
    )

    print("-" * 100)

    for row in retrieval_ranking:

        print(
            f"{row['retrieval_rank']:<6}"
            f"{row['model']:<22}"
            f"{row['recall_at_1']:<12.4f}"
            f"{row['recall_at_5']:<12.4f}"
            f"{row['recall_at_10']:<12.4f}"
        )

    print()
    print("=" * 100)
    print(
        "UNIFIED BENCHMARK COMPLETE"
    )
    print("=" * 100)

    foundation_models = sorted(
        {
            record["model_id"]
            for record in records
            if record["task"]
            != "wsi_mil_prototype"
        }
    )

    print(
        "Foundation models evaluated:",
        len(foundation_models),
    )

    print(
        "Canonical models:",
        ", ".join(
            foundation_models
        ),
    )

    print(
        "Benchmark records:",
        len(records),
    )

    print()
    print(
        "JSON:",
        json_path,
    )

    print(
        "CSV :",
        csv_path,
    )

    print("=" * 100)


if __name__ == "__main__":
    main()