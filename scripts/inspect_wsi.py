from __future__ import annotations

import argparse
import json

from pathoverse.wsi import (
    WSIReader,
    level_mpp,
    level_objective_power,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a whole-slide image."
    )

    parser.add_argument(
        "--slide",
        required=True,
        help="Path to WSI file.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metadata as JSON.",
    )

    args = parser.parse_args()

    with WSIReader(args.slide) as reader:

        metadata = reader.metadata

        if args.json:
            print(
                json.dumps(
                    metadata.to_dict(),
                    indent=2,
                )
            )
            return

        print("=" * 78)
        print("PATHOVERSE AI — WSI INSPECTION")
        print("=" * 78)

        print("\nSLIDE")
        print("-" * 78)

        print(f"Slide ID       : {metadata.slide_id}")
        print(f"Filename       : {metadata.filename}")
        print(f"Format         : {metadata.format}")
        print(f"Vendor         : {metadata.vendor}")

        print("\nDIMENSIONS")
        print("-" * 78)

        print(
            f"Level 0        : "
            f"{metadata.width} × {metadata.height}"
        )

        print(
            f"Megapixels     : "
            f"{metadata.megapixels:,.2f}"
        )

        print(
            f"Gigapixel WSI  : "
            f"{metadata.is_gigapixel}"
        )

        print("\nPYRAMID")
        print("-" * 78)

        print(
            f"Level count    : "
            f"{metadata.level_count}"
        )

        for level in metadata.levels:

            mpp_x, mpp_y = level_mpp(
                reader,
                level.level,
            )

            objective = level_objective_power(
                reader,
                level.level,
            )

            print(
                f"Level {level.level:<2}       : "
                f"{level.width} × {level.height} "
                f"| downsample={level.downsample:.4f}"
            )

            print(
                f"                 MPP="
                f"{mpp_x if mpp_x is not None else 'N/A'} × "
                f"{mpp_y if mpp_y is not None else 'N/A'}"
            )

            print(
                f"                 objective="
                f"{objective if objective is not None else 'N/A'}"
            )

        print("\nMETADATA")
        print("-" * 78)

        print(
            f"MPP X          : "
            f"{metadata.mpp_x}"
        )

        print(
            f"MPP Y          : "
            f"{metadata.mpp_y}"
        )

        print(
            f"Objective Power : "
            f"{metadata.objective_power}"
        )

        print("\nSTATUS")
        print("-" * 78)
        print("✓ WSI opened successfully")
        print("✓ Metadata extracted successfully")
        print("✓ Pyramid levels inspected")

        print("=" * 78)


if __name__ == "__main__":
    main()