from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


class PathoVerseMLflowTracker:
    """
    MLflow experiment tracker for PathoVerse AI.

    Uses a local SQLite backend so the experiment store is
    compatible with current MLflow versions.
    """

    def __init__(
        self,
        experiment_name: str = "PathoVerse",
        tracking_uri: str = "sqlite:///mlflow.db",
    ):

        mlflow.set_tracking_uri(
            tracking_uri
        )

        mlflow.set_experiment(
            experiment_name
        )

        self.experiment_name = (
            experiment_name
        )

        self.tracking_uri = (
            tracking_uri
        )

    # ========================================================
    # SAFE NUMBER
    # ========================================================

    @staticmethod
    def _safe_number(
        value,
    ):

        if value is None:
            return None

        try:

            number = float(value)

            if number != number:
                return None

            return number

        except (
            TypeError,
            ValueError,
        ):

            return None

    # ========================================================
    # LOG RECORD
    # ========================================================

    def log_record(
        self,
        record: dict[str, Any],
        artifact_paths: list[
            str | Path
        ] | None = None,
    ):

        model_name = record[
            "model"
        ]

        task = record[
            "task"
        ]

        run_name = (
            f"{model_name}_{task}"
        )

        with mlflow.start_run(
            run_name=run_name,
        ):

            # ------------------------------------------------
            # PARAMETERS
            # ------------------------------------------------

            params = {

                "model": model_name,

                "model_id": record[
                    "model_id"
                ],

                "task": task,

                "embedding_dimension": (
                    record[
                        "embedding_dimension"
                    ]
                ),

                "dataset": record[
                    "dataset"
                ],

                "split": record[
                    "split"
                ],

                "samples": record[
                    "samples"
                ],

                "status": record[
                    "status"
                ],

            }

            clean_params = {
                key: str(value)
                for key, value
                in params.items()
                if value is not None
            }

            if clean_params:

                mlflow.log_params(
                    clean_params
                )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            metric_fields = [

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

            ]

            metrics = {}

            for field in metric_fields:

                value = self._safe_number(
                    record.get(field)
                )

                if value is not None:

                    metrics[field] = value

            if metrics:

                mlflow.log_metrics(
                    metrics
                )

            # ------------------------------------------------
            # TAGS
            # ------------------------------------------------

            tags = {

                "project": "PathoVerse AI",

                "task": task,

                "model": model_name,

                "status": record[
                    "status"
                ],

            }

            metadata = record.get(
                "metadata"
            )

            if isinstance(
                metadata,
                dict,
            ):

                trained = metadata.get(
                    "trained"
                )

                if trained is not None:

                    tags[
                        "trained"
                    ] = str(trained)

            mlflow.set_tags(
                tags
            )

            # ------------------------------------------------
            # ARTIFACTS
            # ------------------------------------------------

            if artifact_paths:

                for artifact in artifact_paths:

                    artifact = Path(
                        artifact
                    )

                    if not artifact.exists():

                        continue

                    if artifact.is_file():

                        mlflow.log_artifact(
                            str(artifact)
                        )

                    elif artifact.is_dir():

                        mlflow.log_artifacts(
                            str(artifact)
                        )

            # ------------------------------------------------
            # RUN ID
            # ------------------------------------------------

            run_id = (
                mlflow.active_run()
                .info
                .run_id
            )

            return run_id