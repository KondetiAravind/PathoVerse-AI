from __future__ import annotations

import json
from pathlib import Path

from pathoverse.datasets.patchcamelyon import (
    PatchCamelyonDataset,
)


ROOT = Path(
    "data/raw/patchcamelyon"
)

OUTPUT = Path(
    "data/metadata/patchcamelyon/statistics.json"
)


def main():

    print("=" * 72)
    print("PATHOVERSE — PATCHCAMELYON STATISTICS")
    print("=" * 72)

    dataset = PatchCamelyonDataset(
        ROOT
    )

    stats = dataset.statistics()

    print()
    print(
        json.dumps(
            stats,
            indent=2,
        )
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )

    print()
    print(
        f"✓ Statistics saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()