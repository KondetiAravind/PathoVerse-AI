from __future__ import annotations

import json
from pathlib import Path


GROUND_TRUTH_PATH = Path(
    "results/retrieval/ground_truth.json"
)


CATEGORIES = {
    "1": (
        "epithelial_rich",
        "Epithelial-rich"
    ),
    "2": (
        "stroma_collagen_rich",
        "Stroma/collagen-rich"
    ),
    "3": (
        "mixed_epithelium_stroma",
        "Mixed epithelial + stroma"
    ),
}


def parse_tile_ids(value: str) -> list[int]:
    if not value.strip():
        return []

    ids = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        tile_id = int(item)

        if not 0 <= tile_id <= 44:
            raise ValueError(
                f"Invalid tile ID: {tile_id}. "
                "Expected 0-44."
            )

        ids.append(tile_id)

    return sorted(set(ids))


def main() -> None:

    print("=" * 72)
    print("PATHOVERSE — RETRIEVAL GROUND-TRUTH ANNOTATION")
    print("=" * 72)

    with open(GROUND_TRUTH_PATH, "r") as f:
        data = json.load(f)

    print()
    print("Enter tile IDs separated by commas.")
    print("Example: 5,6,8,9,10")
    print()
    print("Use only visually defensible categories.")
    print()

    for number, (category, label) in CATEGORIES.items():

        print("-" * 72)
        print(f"{number}. {label}")
        print("-" * 72)

        value = input(
            "Tile IDs: "
        )

        tile_ids = parse_tile_ids(value)

        data["categories"][category][
            "relevant_tiles"
        ] = tile_ids

        print(
            f"Recorded {len(tile_ids)} tiles."
        )
        print()

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    assigned = []

    for category_data in data["categories"].values():

        assigned.extend(
            category_data["relevant_tiles"]
        )

    duplicates = (
        len(assigned)
        != len(set(assigned))
    )

    if duplicates:

        print(
            "WARNING: Some tiles appear in "
            "multiple categories."
        )

    # ------------------------------------------------------
    # Mark complete
    # ------------------------------------------------------

    data["dataset"][
        "annotation_status"
    ] = "manually_verified"

    with open(
        GROUND_TRUTH_PATH,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )

    print("=" * 72)
    print("GROUND TRUTH SAVED")
    print("=" * 72)
    print(
        GROUND_TRUTH_PATH
    )

    print()

    for category, category_data in (
        data["categories"].items()
    ):

        print(
            f"{category:<30}"
            f"{len(category_data['relevant_tiles']):>4} tiles"
        )


if __name__ == "__main__":
    main()
