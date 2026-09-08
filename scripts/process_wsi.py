from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathoverse.wsi import (
    TissueDetector,
    TileCoordinateEngine,
    WSIReader,
    build_manifest,
    save_manifest,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="PathoVerse AI WSI processing pipeline"
    )

    parser.add_argument(
        "--slide",
        required=True,
        help="Path to WSI file",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/wsi",
        help="Output directory",
    )

    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--min-tissue",
        type=float,
        default=0.50,
    )

    args = parser.parse_args()

    slide_path = Path(args.slide)

    if not slide_path.exists():
        raise FileNotFoundError(
            f"Slide not found: {slide_path}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("PATHOVERSE AI — WSI PROCESSING PIPELINE")
    print("=" * 72)

    print(f"Slide       : {slide_path}")
    print(f"Output      : {output_dir}")
    print()

    with WSIReader(slide_path) as reader:

        metadata = reader.metadata

        print("WSI")
        print("-" * 72)
        print(f"Format      : {metadata.format}")
        print(f"Vendor      : {metadata.vendor}")
        print(f"Dimensions  : {metadata.width} × {metadata.height}")
        print(f"Levels      : {metadata.level_count}")
        print(f"MPP         : {metadata.mpp_x}, {metadata.mpp_y}")
        print(
            f"Objective   : "
            f"{metadata.objective_power}"
        )
        print()

        metadata_path = (
            output_dir
            / f"{metadata.slide_id}_metadata.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata.to_dict(),
                file,
                indent=2,
            )

        print(
            f"✓ Metadata saved: {metadata_path}"
        )

        thumbnail = reader.read_thumbnail(
            args.thumbnail_width,
            args.thumbnail_width,
        )

        thumbnail_path = (
            output_dir
            / f"{metadata.slide_id}_thumbnail.jpg"
        )

        thumbnail.save(
            thumbnail_path,
            quality=95,
        )

        print(
            f"✓ Thumbnail saved: {thumbnail_path}"
        )

        detector = TissueDetector(
            thumbnail_width=args.thumbnail_width,
        )

        tissue_result = detector.detect(
            reader
        )

        mask_path = (
            output_dir
            / f"{metadata.slide_id}_tissue_mask.png"
        )

        tissue_result.save(mask_path)

        print(
            f"✓ Tissue mask saved: {mask_path}"
        )

        print(
            f"  Tissue ratio: "
            f"{tissue_result.tissue_ratio:.4f}"
        )

        tiler = TileCoordinateEngine(
            tile_size=args.tile_size,
            stride=args.stride,
            min_tissue_ratio=args.min_tissue,
        )

        tiles = tiler.generate(
            reader,
            tissue_result.mask,
        )

        print()
        print("TILING")
        print("-" * 72)
        print(
            f"Tile size   : {args.tile_size}"
        )
        print(
            f"Stride      : {args.stride}"
        )
        print(
            f"Min tissue  : {args.min_tissue}"
        )
        print(
            f"Tile count  : {len(tiles)}"
        )

        manifest = build_manifest(
            metadata,
            tiles,
            level=0,
            tile_size=args.tile_size,
            stride=args.stride,
            min_tissue_ratio=args.min_tissue,
        )

        manifest_path = (
            output_dir
            / f"{metadata.slide_id}_tiles.json"
        )

        save_manifest(
            manifest,
            manifest_path,
        )

        print(
            f"✓ Manifest saved: {manifest_path}"
        )

    print()
    print("=" * 72)
    print("✓ WSI PROCESSING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()