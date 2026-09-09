from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pathoverse.api.dependencies import get_results_root
from pathoverse.api.services.model_service import (
    normalize_model_id,
)


MODEL_RESULT_FILES = {
    "vit-b-16": "vit_b_16_classification.json",
    "gigapath-flash": "gigapath_flash_classification.json",
    "conch": "conch_classification.json",
}


class AnalysisService:
    """Read-only access to completed PathoVerse analyses."""

    def __init__(self) -> None:
        self.results_root = get_results_root()

    def get_classification(
        self,
        model: str,
    ) -> dict[str, Any]:
        model = normalize_model_id(model)

        filename = MODEL_RESULT_FILES.get(model)

        if filename is None:
            raise FileNotFoundError(
                f"No classification artifact mapping "
                f"for '{model}'."
            )

        path = (
            self.results_root
            / "classification"
            / filename
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Classification artifact not found: {path}"
            )

        return self._load_json(path)

    def get_mil(
        self,
        slide_id: str,
        model: str,
    ) -> dict[str, Any]:
        model = normalize_model_id(model)

        model_name = model.replace("-", "_")

        path = (
            self.results_root
            / "mil"
            / f"{slide_id}_{model_name}_mil.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"MIL artifact not found: {path}"
            )

        return self._load_json(path)

    def get_heatmap_path(
        self,
        slide_id: str,
        model: str,
    ) -> Path:
        model = normalize_model_id(model)

        model_name = model.replace("-", "_")

        path = (
            self.results_root
            / "mil"
            / (
                f"{slide_id}_{model_name}"
                "_attention_heatmap.png"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Heatmap not found: {path}"
            )

        return path

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict[str, Any]:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)


analysis_service = AnalysisService()