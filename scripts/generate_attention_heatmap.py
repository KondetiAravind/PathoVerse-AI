from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# PATHS
# ============================================================

WSI_THUMBNAIL_DIR = Path(
    "data/processed/wsi"
)

RESULTS_DIR = Path(
    "results/mil"
)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate WSI attention heatmap "
            "from MIL attention weights."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "vit_b_16",
            "gigapath_flash",
            "conch",
        ],
    )

    parser.add_argument(
        "--slide-id",
        default="CMU-1-Small-Region",
    )

    args = parser.parse_args()

    # ========================================================
    # Input
    # ========================================================

    result_path = (
        RESULTS_DIR
        / (
            f"{args.slide_id}_"
            f"{args.model}_mil.json"
        )
    )

    thumbnail_path = (
        WSI_THUMBNAIL_DIR
        / (
            f"{args.slide_id}_thumbnail.jpg"
        )
    )

    if not result_path.exists():

        raise FileNotFoundError(
            f"MIL result not found: "
            f"{result_path}"
        )

    if not thumbnail_path.exists():

        raise FileNotFoundError(
            f"WSI thumbnail not found: "
            f"{thumbnail_path}"
        )

    with open(
        result_path,
        "r",
    ) as f:

        result = json.load(
            f
        )

    # ========================================================
    # Load thumbnail
    # ========================================================

    thumbnail = Image.open(
        thumbnail_path
    ).convert(
        "RGB"
    )

    width, height = thumbnail.size

    # ========================================================
    # Determine coordinate scaling
    # ========================================================

    all_tiles = result[
        "attention"
    ][
        "all_tiles"
    ]

    max_x = max(
        tile["x"] + tile["width"]
        for tile in all_tiles
    )

    max_y = max(
        tile["y"] + tile["height"]
        for tile in all_tiles
    )

    scale_x = (
        width / max_x
    )

    scale_y = (
        height / max_y
    )

    # ========================================================
    # Normalize attention
    # ========================================================

    attention = np.asarray(
        [
            tile["attention"]
            for tile in all_tiles
        ],
        dtype=np.float32,
    )

    min_attention = float(
        attention.min()
    )

    max_attention = float(
        attention.max()
    )

    if (
        max_attention
        - min_attention
    ) > 1e-12:

        normalized = (
            attention
            - min_attention
        ) / (
            max_attention
            - min_attention
        )

    else:

        normalized = np.zeros_like(
            attention
        )

    # ========================================================
    # Draw overlay
    # ========================================================

    overlay = Image.new(
        "RGBA",
        thumbnail.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(
        overlay,
        "RGBA",
    )

    for tile, value in zip(
        all_tiles,
        normalized,
    ):

        x1 = int(
            tile["x"]
            * scale_x
        )

        y1 = int(
            tile["y"]
            * scale_y
        )

        x2 = int(
            (
                tile["x"]
                + tile["width"]
            )
            * scale_x
        )

        y2 = int(
            (
                tile["y"]
                + tile["height"]
            )
            * scale_y
        )

        # Heat intensity:
        # low attention -> transparent
        # high attention -> stronger overlay
        alpha = int(
            40
            + 180 * float(value)
        )

        # Use a simple red intensity map.
        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            fill=(
                255,
                0,
                0,
                alpha,
            ),
        )

    result_image = Image.alpha_composite(
        thumbnail.convert(
            "RGBA"
        ),
        overlay,
    )

    # ========================================================
    # Draw top-k borders
    # ========================================================

    top_tiles = result[
        "attention"
    ][
        "top_tiles"
    ]

    draw = ImageDraw.Draw(
        result_image
    )

    for tile in top_tiles:

        x1 = int(
            tile["x"]
            * scale_x
        )

        y1 = int(
            tile["y"]
            * scale_y
        )

        x2 = int(
            (
                tile["x"]
                + tile["width"]
            )
            * scale_x
        )

        y2 = int(
            (
                tile["y"]
                + tile["height"]
            )
            * scale_y
        )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline=(
                255,
                255,
                255,
                255,
            ),
            width=3,
        )

    # ========================================================
    # Save
    # ========================================================

    output_path = (
        RESULTS_DIR
        / (
            f"{args.slide_id}_"
            f"{args.model}_attention_heatmap.png"
        )
    )

    result_image.convert(
        "RGB"
    ).save(
        output_path,
        quality=95,
    )

    print()
    print("=" * 72)

    print(
        "PATHOVERSE — WSI ATTENTION HEATMAP"
    )

    print("=" * 72)

    print(
        "Slide:",
        args.slide_id,
    )

    print(
        "Model:",
        args.model,
    )

    print(
        "Thumbnail:",
        thumbnail.size,
    )

    print(
        "Attention range:",
        f"{min_attention:.6f}",
        "→",
        f"{max_attention:.6f}",
    )

    print()
    print(
        "Saved:",
        output_path,
    )

    print("=" * 72)


if __name__ == "__main__":
    main()