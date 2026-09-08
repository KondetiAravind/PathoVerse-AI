from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathoverse.tiles import (
    TileExtractor,
    TileQualityChecker,
    save_tile_manifest,
)
from pathoverse.wsi import WSIReader


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract ML-ready tiles from "
            "a PathoVerse WSI manifest."
        )
    )

    parser.add_argument(
        "--slide",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/tiles",
    )

    parser.add_argument(
        "--output-manifest",
        default=(
            "data/metadata/tiles/"
            "extracted_tiles.json"
        ),
    )

    parser.add_argument(
        "--level",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--min-tissue",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--min-std",
        type=float,
        default=8.0,
    )

    args = parser.parse_args()

    slide_path = Path(args.slide)
    manifest_path = Path(args.manifest)

    if not slide_path.exists():
        raise FileNotFoundError(
            f"Slide not found: {slide_path}"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: "
            f"{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    tiles = manifest.get(
        "tiles",
        [],
    )

    print("=" * 72)
    print(
        "PATHOVERSE AI — TILE EXTRACTION"
    )
    print("=" * 72)

    print(
        f"Slide          : {slide_path}"
    )
    print(
        f"Input manifest : {manifest_path}"
    )
    print(
        f"Input tiles    : {len(tiles)}"
    )
    print(
        f"Output         : {args.output_dir}"
    )
    print()

    quality_checker = TileQualityChecker(
        min_tissue_ratio=args.min_tissue,
        min_std=args.min_std,
    )

    extractor = TileExtractor(
        args.output_dir,
        level=args.level,
        quality_checker=quality_checker,
        image_format="png",
    )

    with WSIReader(slide_path) as reader:

        extracted, rejected = (
            extractor.extract(
                reader,
                tiles,
            )
        )

    save_tile_manifest(
        extracted,
        rejected,
        args.output_manifest,
    )

    print("RESULT")
    print("-" * 72)

    print(
        f"Extracted      : {len(extracted)}"
    )

    print(
        f"Rejected       : {len(rejected)}"
    )

    total = (
        len(extracted)
        + len(rejected)
    )

    if total:
        pass_rate = (
            len(extracted) / total
        ) * 100
    else:
        pass_rate = 0.0

    print(
        f"Pass rate      : {pass_rate:.2f}%"
    )

    print()
    print(
        f"✓ Manifest saved: "
        f"{args.output_manifest}"
    )

    print()
    print("=" * 72)
    print(
        "✓ TILE EXTRACTION COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()