from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .base import BasePathologyDataset, DatasetInfo
from .registry import register_dataset


class PatchCamelyonTorchDataset(Dataset):
    """
    PyTorch dataset wrapper for PatchCamelyon HDF5 data.

    PCam stores:
        x -> image tensors
        y -> binary labels
    """

    def __init__(
        self,
        image_file: str | Path,
        label_file: str | Path,
        transform=None,
    ):
        self.image_file = Path(image_file)
        self.label_file = Path(label_file)
        self.transform = transform

        if not self.image_file.exists():
            raise FileNotFoundError(
                f"Missing image file: {self.image_file}"
            )

        if not self.label_file.exists():
            raise FileNotFoundError(
                f"Missing label file: {self.label_file}"
            )

        with h5py.File(self.image_file, "r") as f:
            self.length = len(f["x"])

        with h5py.File(self.label_file, "r") as f:
            label_length = len(f["y"])

        if self.length != label_length:
            raise ValueError(
                f"Image/label mismatch: "
                f"{self.length} vs {label_length}"
            )

        self._images = None
        self._labels = None

    def _open_files(self):
        if self._images is None:
            self._images = h5py.File(
                self.image_file,
                "r",
            )["x"]

        if self._labels is None:
            self._labels = h5py.File(
                self.label_file,
                "r",
            )["y"]

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        self._open_files()

        image = self._images[index]
        label = self._labels[index]

        image = Image.fromarray(
            np.asarray(image)
        )

        if self.transform is not None:
            image = self.transform(image)

        label = int(
            np.asarray(label).reshape(-1)[0]
        )

        return image, label

    def __del__(self):
        try:
            if self._images is not None:
                self._images.file.close()
        except Exception:
            pass

        try:
            if self._labels is not None:
                self._labels.file.close()
        except Exception:
            pass


@register_dataset("patchcamelyon")
class PatchCamelyonDataset(BasePathologyDataset):
    """
    PatchCamelyon dataset adapter.

    Expected files:

        data/raw/patchcamelyon/
            camelyonpatch_level_2_split_train_x.h5
            camelyonpatch_level_2_split_train_y.h5
            camelyonpatch_level_2_split_valid_x.h5
            camelyonpatch_level_2_split_valid_y.h5
            camelyonpatch_level_2_split_test_x.h5
            camelyonpatch_level_2_split_test_y.h5
    """

    SPLITS = {
        "train": (
            "camelyonpatch_level_2_split_train_x.h5",
            "camelyonpatch_level_2_split_train_y.h5",
        ),
        "validation": (
            "camelyonpatch_level_2_split_valid_x.h5",
            "camelyonpatch_level_2_split_valid_y.h5",
        ),
        "test": (
            "camelyonpatch_level_2_split_test_x.h5",
            "camelyonpatch_level_2_split_test_y.h5",
        ),
    }

    def __init__(
        self,
        root_dir: str | Path,
    ):
        super().__init__(root_dir)

    @property
    def name(self) -> str:
        return "patchcamelyon"

    def _split_paths(self, split: str):
        if split not in self.SPLITS:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Expected one of {list(self.SPLITS)}"
            )

        image_name, label_name = self.SPLITS[split]

        return (
            self.root_dir / image_name,
            self.root_dir / label_name,
        )

    def download(self, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Use scripts/download_patchcamelyon.py "
            "for PCam acquisition."
        )

    def get_split(
        self,
        split: str,
        transform=None,
    ):
        image_file, label_file = self._split_paths(split)

        return PatchCamelyonTorchDataset(
            image_file=image_file,
            label_file=label_file,
            transform=transform,
        )

    def validate(self) -> Dict[str, Any]:
        errors = []
        splits = {}

        for split in self.SPLITS:
            try:
                image_file, label_file = self._split_paths(split)

                if not image_file.exists():
                    errors.append(
                        f"Missing image file: {image_file}"
                    )

                if not label_file.exists():
                    errors.append(
                        f"Missing label file: {label_file}"
                    )

                if image_file.exists() and label_file.exists():
                    with h5py.File(image_file, "r") as image_h5:
                        image_count = len(image_h5["x"])
                        image_shape = tuple(
                            image_h5["x"].shape
                        )

                    with h5py.File(label_file, "r") as label_h5:
                        label_count = len(label_h5["y"])

                    if image_count != label_count:
                        errors.append(
                            f"{split}: image/label count mismatch "
                            f"{image_count} vs {label_count}"
                        )

                    splits[split] = {
                        "images": image_count,
                        "labels": label_count,
                        "image_shape": image_shape,
                    }

            except Exception as exc:
                errors.append(
                    f"{split}: validation error: {exc}"
                )

        return {
            "dataset": self.name,
            "valid": len(errors) == 0,
            "status": (
                "valid"
                if len(errors) == 0
                else "invalid"
            ),
            "splits": splits,
            "errors": errors,
        }

    def statistics(self) -> Dict[str, Any]:
        validation = self.validate()

        class_counts = {}

        for split in self.SPLITS:
            image_file, label_file = self._split_paths(split)

            if not label_file.exists():
                continue

            with h5py.File(label_file, "r") as f:
                labels = np.asarray(f["y"]).reshape(-1)

            unique, counts = np.unique(
                labels,
                return_counts=True,
            )

            class_counts[split] = {
                str(int(label)): int(count)
                for label, count in zip(
                    unique,
                    counts,
                )
            }

        return {
            "dataset": self.name,
            "validation": validation,
            "class_counts": class_counts,
        }

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="patchcamelyon",
            description=(
                "Binary metastatic tissue classification "
                "dataset based on histopathology patches."
            ),
            num_classes=2,
            class_names=[
                "normal",
                "metastatic",
            ],
            image_shape=(96, 96, 3),
            split_names=[
                "train",
                "validation",
                "test",
            ],
            root_dir=self.root_dir,
        )