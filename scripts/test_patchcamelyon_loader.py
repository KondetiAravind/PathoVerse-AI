from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from pathoverse.datasets.patchcamelyon import (
    PatchCamelyonDataset,
)


ROOT = Path(
    "data/raw/patchcamelyon"
)


def main():

    print("=" * 72)
    print("PATHOVERSE — PATCHCAMELYON LOADER TEST")
    print("=" * 72)

    dataset = PatchCamelyonDataset(
        ROOT
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for split in [
        "train",
        "validation",
        "test",
    ]:

        split_dataset = dataset.get_split(
            split,
            transform=transform,
        )

        print()
        print(
            f"{split:<12}: "
            f"{len(split_dataset):>8} samples"
        )

        loader = DataLoader(
            split_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )

        images, labels = next(
            iter(loader)
        )

        print(
            "  image shape:",
            tuple(images.shape),
        )

        print(
            "  image dtype:",
            images.dtype,
        )

        print(
            "  image range:",
            float(images.min()),
            "to",
            float(images.max()),
        )

        print(
            "  labels:",
            labels.tolist(),
        )

        assert images.ndim == 4
        assert images.shape[1:] == (
            3,
            96,
            96,
        )

        assert labels.ndim == 1

        assert torch.isfinite(
            images
        ).all()

    print()
    print(
        "✓ PATCHCAMELYON LOADER TEST PASSED"
    )


if __name__ == "__main__":
    main()