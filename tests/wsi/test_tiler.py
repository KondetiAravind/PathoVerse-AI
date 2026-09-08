from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pathoverse.wsi import (
    WSIReader,
    level_mpp,
    level_objective_power,
    select_level,
)


def fake_slide():
    slide = MagicMock()

    slide.properties = {
        "openslide.vendor": "Aperio",
        "openslide.mpp-x": "0.25",
        "openslide.mpp-y": "0.25",
        "openslide.objective-power": "40",
    }

    slide.dimensions = (
        50000,
        40000,
    )

    slide.level_count = 4

    slide.level_dimensions = (
        (50000, 40000),
        (25000, 20000),
        (12500, 10000),
        (6250, 5000),
    )

    slide.level_downsamples = (
        1.0,
        2.0,
        4.0,
        8.0,
    )

    slide.get_best_level_for_downsample.return_value = 2

    return slide


def open_reader(tmp_path):
    slide_path = tmp_path / "sample.svs"
    slide_path.write_bytes(b"fake")

    slide = fake_slide()

    patches = [
        patch(
            "pathoverse.wsi.reader.openslide.OpenSlide.detect_format",
            return_value="Aperio",
        ),
        patch(
            "pathoverse.wsi.reader.openslide.OpenSlide",
            return_value=slide,
        ),
    ]

    for patcher in patches:
        patcher.start()

    reader = WSIReader(slide_path)

    return reader, patches


def close_reader(reader, patches):
    reader.close()

    for patcher in patches:
        patcher.stop()


def test_explicit_level(tmp_path):
    reader, patches = open_reader(tmp_path)

    try:
        assert (
            select_level(
                reader,
                level=2,
            )
            == 2
        )
    finally:
        close_reader(reader, patches)


def test_target_downsample(tmp_path):
    reader, patches = open_reader(tmp_path)

    try:
        assert (
            select_level(
                reader,
                target_downsample=4.0,
            )
            == 2
        )
    finally:
        close_reader(reader, patches)


def test_default_level(tmp_path):
    reader, patches = open_reader(tmp_path)

    try:
        assert (
            select_level(reader)
            == 0
        )
    finally:
        close_reader(reader, patches)


def test_level_mpp(tmp_path):
    reader, patches = open_reader(tmp_path)

    try:
        mpp_x, mpp_y = level_mpp(
            reader,
            2,
        )

        assert mpp_x == 1.0
        assert mpp_y == 1.0
    finally:
        close_reader(reader, patches)


def test_level_objective(tmp_path):
    reader, patches = open_reader(tmp_path)

    try:
        objective = level_objective_power(
            reader,
            2,
        )

        assert objective == 10.0
    finally:
        close_reader(reader, patches)


def test_invalid_level(tmp_path):
    reader, patches = open_reader(tmp_path)

    try:
        with pytest.raises(ValueError):
            select_level(
                reader,
                level=99,
            )
    finally:
        close_reader(reader, patches)


def test_conflicting_level_arguments(tmp_path):
    reader, patches = open_reader(tmp_path)

    try:
        with pytest.raises(ValueError):
            select_level(
                reader,
                level=1,
                target_downsample=2.0,
            )
    finally:
        close_reader(reader, patches)