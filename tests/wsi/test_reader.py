from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from pathoverse.wsi import WSIReader


def create_fake_slide():
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

    slide.read_region.return_value = Image.new(
        "RGBA",
        (256, 256),
    )

    slide.get_thumbnail.return_value = Image.new(
        "RGB",
        (512, 512),
    )

    slide.get_best_level_for_downsample.return_value = 2

    return slide

def test_reader_opens_slide(tmp_path: Path):
    slide_path = tmp_path / "sample.svs"
    slide_path.write_bytes(b"fake")

    fake_slide = create_fake_slide()

    with patch(
        "pathoverse.wsi.reader.openslide.OpenSlide"
    ) as mock_openslide:

        mock_openslide.detect_format.return_value = "Aperio"
        mock_openslide.return_value = fake_slide

        reader = WSIReader(slide_path)

        assert reader.is_open
        assert reader.format == "Aperio"
        assert reader.dimensions == (50000, 40000)

def test_metadata(tmp_path: Path):
    slide_path = tmp_path / "sample.svs"
    slide_path.write_bytes(b"fake")

    fake_slide = create_fake_slide()

    with patch(
        "pathoverse.wsi.reader.openslide.OpenSlide.detect_format",
        return_value="Aperio",
    ), patch(
        "pathoverse.wsi.reader.openslide.OpenSlide",
        return_value=fake_slide,
    ):

        reader = WSIReader(slide_path)

        metadata = reader.metadata

        assert metadata.slide_id == "sample"
        assert metadata.vendor == "Aperio"
        assert metadata.mpp_x == 0.25
        assert metadata.mpp_y == 0.25
        assert metadata.objective_power == 40.0
        assert metadata.is_gigapixel

        reader.close()


def test_read_region(tmp_path: Path):
    slide_path = tmp_path / "sample.svs"
    slide_path.write_bytes(b"fake")

    fake_slide = create_fake_slide()

    with patch(
        "pathoverse.wsi.reader.openslide.OpenSlide.detect_format",
        return_value="Aperio",
    ), patch(
        "pathoverse.wsi.reader.openslide.OpenSlide",
        return_value=fake_slide,
    ):

        reader = WSIReader(slide_path)

        image = reader.read_region(
            1000,
            2000,
            256,
            256,
            level=2,
        )

        assert image.size == (
            256,
            256,
        )

        assert image.mode == "RGB"

        fake_slide.read_region.assert_called_once_with(
            (1000, 2000),
            2,
            (256, 256),
        )

        reader.close()


def test_invalid_level(tmp_path: Path):
    slide_path = tmp_path / "sample.svs"
    slide_path.write_bytes(b"fake")

    fake_slide = create_fake_slide()

    with patch(
        "pathoverse.wsi.reader.openslide.OpenSlide.detect_format",
        return_value="Aperio",
    ), patch(
        "pathoverse.wsi.reader.openslide.OpenSlide",
        return_value=fake_slide,
    ):

        reader = WSIReader(slide_path)

        with pytest.raises(ValueError):
            reader.read_region(
                0,
                0,
                256,
                256,
                level=99,
            )

        reader.close()


def test_best_level(tmp_path: Path):
    slide_path = tmp_path / "sample.svs"
    slide_path.write_bytes(b"fake")

    fake_slide = create_fake_slide()

    with patch(
        "pathoverse.wsi.reader.openslide.OpenSlide.detect_format",
        return_value="Aperio",
    ), patch(
        "pathoverse.wsi.reader.openslide.OpenSlide",
        return_value=fake_slide,
    ):

        reader = WSIReader(slide_path)

        level = reader.best_level_for_downsample(
            4.0
        )

        assert level == 2

        reader.close()