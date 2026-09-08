from __future__ import annotations

import numpy as np
import h5py
import pytest

from pathoverse.datasets.patchcamelyon import (
    PatchCamelyonDataset,
)


def create_fake_pcam(
    root,
    split="train",
    count=12,
):

    image_file = (
        root
        / f"camelyonpatch_level_2_split_{split}_x.h5"
    )

    label_file = (
        root
        / f"camelyonpatch_level_2_split_{split}_y.h5"
    )

    images = np.random.randint(
        0,
        256,
        size=(count, 96, 96, 3),
        dtype=np.uint8,
    )

    labels = np.arange(
        count
    ) % 2

    with h5py.File(
        image_file,
        "w",
    ) as f:
        f.create_dataset(
            "x",
            data=images,
        )

    with h5py.File(
        label_file,
        "w",
    ) as f:
        f.create_dataset(
            "y",
            data=labels,
        )


def test_dataset_info(tmp_path):

    dataset = PatchCamelyonDataset(
        tmp_path
    )

    info = dataset.info()

    assert info.name == "patchcamelyon"
    assert info.num_classes == 2
    assert info.class_names == [
        "normal",
        "metastatic",
    ]
    assert info.image_shape == (
        96,
        96,
        3,
    )


def test_missing_dataset(tmp_path):

    dataset = PatchCamelyonDataset(
        tmp_path
    )

    result = dataset.validate()

    assert result["valid"] is False
    assert result["status"] == "invalid"
    assert len(result["errors"]) > 0


def test_fake_dataset_validation(tmp_path):

    for split in [
        "train",
        "validation",
        "test",
    ]:
        create_fake_pcam(
            tmp_path,
            split=(
                "valid"
                if split == "validation"
                else split
            ),
        )

    dataset = PatchCamelyonDataset(
        tmp_path
    )

    result = dataset.validate()

    assert result["valid"] is True

    assert result["splits"]["train"]["images"] == 12
    assert result["splits"]["validation"]["images"] == 12
    assert result["splits"]["test"]["images"] == 12


def test_fake_dataset_loader(tmp_path):

    create_fake_pcam(
        tmp_path,
        split="train",
        count=5,
    )

    dataset = PatchCamelyonDataset(
        tmp_path
    )

    split = dataset.get_split(
        "train"
    )

    assert len(split) == 5

    image, label = split[0]

    assert image.size == (
        96,
        96,
    )

    assert label in [0, 1]


def test_invalid_split(tmp_path):

    dataset = PatchCamelyonDataset(
        tmp_path
    )

    with pytest.raises(ValueError):
        dataset.get_split(
            "invalid"
        )