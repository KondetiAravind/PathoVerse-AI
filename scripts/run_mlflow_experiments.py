from __future__ import annotations

import json
from pathlib import Path

from pathoverse.benchmark.mlflow_tracker import (
    PathoVerseMLflowTracker,
)


BENCHMARK_PATH = Path(
    "results/unified/foundation_model_benchmark.json"
)


def main():

    if not BENCHMARK_PATH.exists():

        raise FileNotFoundError(
            "Unified benchmark does not exist. "
            "Run build_unified_benchmark.py first."
        )

    with open(
        BENCHMARK_PATH,
        "r",
    ) as f:

        benchmark = json.load(
            f
        )

    tracker = PathoVerseMLflowTracker(
        experiment_name="PathoVerse",
        tracking_uri="sqlite:///mlflow.db",
    )

    records = benchmark[
        "records"
    ]

    print()
    print("=" * 88)

    print(
        "PATHOVERSE — MLFLOW EXPERIMENT TRACKING"
    )

    print("=" * 88)

    successful = 0

    for record in records:

        print(
            f"Logging: "
            f"{record['model']} "
            f"| {record['task']}"
        )

        tracker.log_record(
            record
        )

        successful += 1

    print()
    print(
        f"Logged experiments: {successful}"
    )

    print(
        "Tracking database: ./mlflow.db"
    )

    print()
    print(
        "Start MLflow UI with:"
    )

    print(
        "mlflow ui"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()