from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class WSILevel:
    """
    Metadata describing one resolution level of a whole-slide image.
    """

    level: int
    width: int
    height: int
    downsample: float

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WSIMetadata:
    """
    Standardized metadata representation for a whole-slide image.
    """

    slide_id: str
    filename: str
    path: str

    format: Optional[str]

    vendor: Optional[str]

    width: int
    height: int

    level_count: int

    levels: tuple[WSILevel, ...]

    mpp_x: Optional[float]
    mpp_y: Optional[float]

    objective_power: Optional[float]

    properties: Dict[str, str]

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    @property
    def is_gigapixel(self) -> bool:
        return self.width * self.height >= 1_000_000_000

    @property
    def level0(self) -> WSILevel:
        return self.levels[0]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)

        result["levels"] = [
            level.to_dict()
            for level in self.levels
        ]

        result["dimensions"] = list(self.dimensions)

        return result

    @classmethod
    def from_slide(
        cls,
        slide,
        path: str | Path,
        slide_format: Optional[str] = None,
    ) -> "WSIMetadata":

        path = Path(path)

        properties = dict(slide.properties)

        vendor = properties.get(
            "openslide.vendor"
        )

        mpp_x = _safe_float(
            properties.get("openslide.mpp-x")
        )

        mpp_y = _safe_float(
            properties.get("openslide.mpp-y")
        )

        objective_power = _safe_float(
            properties.get("openslide.objective-power")
        )

        levels = tuple(
            WSILevel(
                level=index,
                width=int(dimensions[0]),
                height=int(dimensions[1]),
                downsample=float(
                    slide.level_downsamples[index]
                ),
            )
            for index, dimensions
            in enumerate(slide.level_dimensions)
        )

        width, height = slide.dimensions

        return cls(
            slide_id=path.stem,
            filename=path.name,
            path=str(path.resolve()),
            format=slide_format,
            vendor=vendor,
            width=int(width),
            height=int(height),
            level_count=int(slide.level_count),
            levels=levels,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            objective_power=objective_power,
            properties=properties,
        )


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None