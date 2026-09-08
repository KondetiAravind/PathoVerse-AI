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
    "data/metadata/patchcamelyon/validation.json"
)


def main():

    print("=" * 72)
    print("PATHOVERSE — PATCHCAMELYON VALIDATION")
    print("=" * 72)

    dataset = PatchCamelyonDataset(
        ROOT
    )

    result = dataset.validate()

    print()
    print(
        json.dumps(
            result,
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
            result,
            f,
            indent=2,
        )

    if not result["valid"]:
        raise RuntimeError(
            "PatchCamelyon validation failed."
        )

    print()
    print(
        f"✓ Validation saved: {OUTPUT}"
    )
    print(
        "✓ PATCHCAMELYON VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()