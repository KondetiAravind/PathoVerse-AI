from pathlib import Path

from pathoverse.wsi.manifest import (
    build_manifest,
    save_manifest,
)
from pathoverse.wsi.metadata import (
    WSILevel,
    WSIMetadata,
)
from pathoverse.wsi.tiler import TileCoordinate


def create_metadata():
    return WSIMetadata(
        slide_id="test_slide",
        filename="test.svs",
        path="/tmp/test.svs",
        format="Aperio",
        vendor="aperio",
        width=1000,
        height=800,
        level_count=2,
        levels=(
            WSILevel(
                level=0,
                width=1000,
                height=800,
                downsample=1.0,
            ),
            WSILevel(
                level=1,
                width=500,
                height=400,
                downsample=2.0,
            ),
        ),
        mpp_x=0.25,
        mpp_y=0.25,
        objective_power=40.0,
        properties={},
    )


def test_build_manifest():
    metadata = create_metadata()

    tiles = [
        TileCoordinate(
            tile_id=0,
            x=0,
            y=0,
            width=224,
            height=224,
            tissue_ratio=0.85,
        )
    ]

    manifest = build_manifest(
        metadata,
        tiles,
        level=0,
        tile_size=224,
        stride=224,
        min_tissue_ratio=0.5,
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["slide"]["slide_id"] == "test_slide"
    assert manifest["statistics"]["tile_count"] == 1
    assert manifest["tiles"][0]["tissue_ratio"] == 0.85


def test_save_manifest(tmp_path: Path):
    metadata = create_metadata()

    manifest = build_manifest(
        metadata,
        [],
        level=0,
        tile_size=224,
        stride=224,
        min_tissue_ratio=0.5,
    )

    output = tmp_path / "manifest.json"

    save_manifest(
        manifest,
        output,
    )

    assert output.exists()