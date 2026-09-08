from __future__ import annotations

from typing import Optional

from .reader import WSIReader


def select_level(
    reader: WSIReader,
    *,
    level: Optional[int] = None,
    target_downsample: Optional[float] = None,
) -> int:
    """
    Select a WSI pyramid level.

    Priority:
        1. Explicit level
        2. Target downsample
        3. Level 0
    """

    if level is not None and target_downsample is not None:
        raise ValueError(
            "Specify either level or target_downsample, not both."
        )

    if level is not None:
        if level < 0 or level >= reader.level_count:
            raise ValueError(
                f"Invalid level {level}. "
                f"Valid range: "
                f"0 to {reader.level_count - 1}."
            )

        return level

    if target_downsample is not None:
        if target_downsample <= 0:
            raise ValueError(
                "target_downsample must be greater than zero."
            )

        return reader.best_level_for_downsample(
            target_downsample
        )

    return 0


def level_mpp(
    reader: WSIReader,
    level: int,
) -> tuple[Optional[float], Optional[float]]:
    """
    Calculate effective microns-per-pixel at a given level.

    Level-0 MPP comes from slide metadata.
    Higher-level MPP is:

        level_MPP = level0_MPP × downsample
    """

    if level < 0 or level >= reader.level_count:
        raise ValueError(
            f"Invalid level {level}."
        )

    if reader.mpp_x is None:
        mpp_x = None
    else:
        mpp_x = (
            reader.mpp_x
            * reader.level_downsamples[level]
        )

    if reader.mpp_y is None:
        mpp_y = None
    else:
        mpp_y = (
            reader.mpp_y
            * reader.level_downsamples[level]
        )

    return mpp_x, mpp_y


def level_objective_power(
    reader: WSIReader,
    level: int,
) -> Optional[float]:
    """
    Estimate objective magnification at a pyramid level.

    If the slide reports an objective power:

        level_objective = base_objective / downsample
    """

    if level < 0 or level >= reader.level_count:
        raise ValueError(
            f"Invalid level {level}."
        )

    if reader.objective_power is None:
        return None

    downsample = reader.level_downsamples[level]

    if downsample <= 0:
        return None

    return (
        reader.objective_power
        / downsample
    )