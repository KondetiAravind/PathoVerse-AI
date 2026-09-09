from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pathoverse.api.dependencies import get_results_root


class BenchmarkService:
    """Access the unified PathoVerse benchmark."""

    def __init__(self) -> None:
        self.path = (
            get_results_root()
            / "unified"
            / "foundation_model_benchmark.json"
        )

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Unified benchmark not found: {self.path}"
            )

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def records(self) -> list[dict[str, Any]]:
        data = self.load()

        records = data.get("records", [])

        if isinstance(records, list):
            return records

        return []

    def by_task(self, task: str) -> list[dict[str, Any]]:
        return [
            record
            for record in self.records()
            if str(record.get("task", "")).lower()
            == task.lower()
        ]


benchmark_service = BenchmarkService()
