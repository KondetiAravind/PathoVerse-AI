from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from PIL import Image

import openslide

from .metadata import WSIMetadata


class WSIReader:
    """
    High-level PathoVerse wrapper around OpenSlide.

    Responsibilities:
    - validate WSI path
    - detect slide format
    - open the slide
    - expose slide metadata
    - expose pyramid information
    - read regions
    - provide safe lifecycle management
    """

    def __init__(
        self,
        slide_path: str | Path,
        *,
        auto_open: bool = True,
    ):
        self.slide_path = Path(slide_path)

        self._slide = None
        self._metadata: Optional[WSIMetadata] = None
        self._format: Optional[str] = None

        if auto_open:
            self.open()

    @property
    def is_open(self) -> bool:
        return self._slide is not None

    @property
    def slide(self):
        if self._slide is None:
            raise RuntimeError(
                "WSI is not open. Call open() first."
            )

        return self._slide

    @property
    def metadata(self) -> WSIMetadata:
        if self._metadata is None:
            raise RuntimeError(
                "WSI metadata is not available. "
                "Open the slide first."
            )

        return self._metadata

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.metadata.dimensions

    @property
    def level_count(self) -> int:
        return self.metadata.level_count

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            level.dimensions
            for level in self.metadata.levels
        )

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return tuple(
            level.downsample
            for level in self.metadata.levels
        )

    @property
    def properties(self) -> Mapping[str, str]:
        return self.slide.properties

    @property
    def vendor(self) -> Optional[str]:
        return self.metadata.vendor

    @property
    def mpp_x(self) -> Optional[float]:
        return self.metadata.mpp_x

    @property
    def mpp_y(self) -> Optional[float]:
        return self.metadata.mpp_y

    @property
    def objective_power(self) -> Optional[float]:
        return self.metadata.objective_power

    @property
    def format(self) -> Optional[str]:
        return self._format

    def open(self) -> None:
        """
        Open and validate the WSI.
        """

        if self.is_open:
            return

        if not self.slide_path.exists():
            raise FileNotFoundError(
                f"WSI file does not exist: {self.slide_path}"
            )

        if not self.slide_path.is_file():
            raise ValueError(
                f"WSI path is not a file: {self.slide_path}"
            )

        self._format = openslide.OpenSlide.detect_format(
            self.slide_path
        )

        if self._format is None:
            raise openslide.OpenSlideUnsupportedFormatError(
                f"OpenSlide does not recognize: "
                f"{self.slide_path}"
            )

        self._slide = openslide.OpenSlide(
            str(self.slide_path)
        )

        self._metadata = WSIMetadata.from_slide(
            self._slide,
            self.slide_path,
            slide_format=self._format,
        )

    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        level: int = 0,
        convert: str = "RGB",
    ) -> Image.Image:
        """
        Read a rectangular region.

        Coordinates x/y are always specified in the
        level-0 reference frame, following OpenSlide's API.
        """

        if not self.is_open:
            raise RuntimeError(
                "WSI is not open."
            )

        self._validate_level(level)

        if width <= 0 or height <= 0:
            raise ValueError(
                "width and height must be positive."
            )

        if convert not in {
            "RGB",
            "RGBA",
            "L",
        }:
            raise ValueError(
                "convert must be one of: RGB, RGBA, L"
            )

        image = self.slide.read_region(
            (int(x), int(y)),
            int(level),
            (int(width), int(height)),
        )

        return image.convert(convert)

    def read_thumbnail(
        self,
        width: int,
        height: int,
    ) -> Image.Image:
        """
        Generate a thumbnail using OpenSlide.
        """

        if not self.is_open:
            raise RuntimeError(
                "WSI is not open."
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                "Thumbnail dimensions must be positive."
            )

        return self.slide.get_thumbnail(
            (int(width), int(height))
        ).convert("RGB")

    def best_level_for_downsample(
        self,
        downsample: float,
    ) -> int:
        """
        Select the OpenSlide pyramid level closest
        to the requested downsample factor.
        """

        if not self.is_open:
            raise RuntimeError(
                "WSI is not open."
            )

        if downsample <= 0:
            raise ValueError(
                "downsample must be greater than zero."
            )

        return self.slide.get_best_level_for_downsample(
            float(downsample)
        )

    def _validate_level(self, level: int) -> None:
        if not isinstance(level, int):
            raise TypeError(
                "level must be an integer."
            )

        if level < 0 or level >= self.level_count:
            raise ValueError(
                f"Invalid level {level}. "
                f"Valid range: "
                f"0 to {self.level_count - 1}."
            )

    def close(self) -> None:
        """
        Close the OpenSlide object.
        """

        if self._slide is not None:
            self._slide.close()

        self._slide = None
        self._metadata = None
        self._format = None

    def __enter__(self) -> "WSIReader":
        if not self.is_open:
            self.open()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass