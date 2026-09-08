from pathlib import Path
import json
import math

from PIL import Image, ImageDraw


MANIFEST_PATH = Path(
    "data/metadata/tiles/extracted_tiles.json"
)

OUTPUT_PATH = Path(
    "results/retrieval/tile_contact_sheet.jpg"
)

THUMB_WIDTH = 180
THUMB_HEIGHT = 180
LABEL_HEIGHT = 30
COLUMNS = 5


def main():
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    tiles = manifest["tiles"]

    rows = math.ceil(
        len(tiles) / COLUMNS
    )

    sheet_width = (
        COLUMNS * THUMB_WIDTH
    )

    sheet_height = (
        rows
        * (THUMB_HEIGHT + LABEL_HEIGHT)
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        color="white",
    )

    draw = ImageDraw.Draw(sheet)

    for index, tile in enumerate(tiles):

        image_path = Path(
            tile["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (THUMB_WIDTH, THUMB_HEIGHT)
        )

        column = index % COLUMNS
        row = index // COLUMNS

        x = column * THUMB_WIDTH
        y = row * (
            THUMB_HEIGHT + LABEL_HEIGHT
        )

        sheet.paste(
            image,
            (x, y),
        )

        label = (
            f"Tile {tile['tile_id']} "
            f"| tissue={tile['tissue_ratio']:.2f}"
        )

        draw.text(
            (x + 5, y + THUMB_HEIGHT + 6),
            label,
            fill="black",
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        OUTPUT_PATH,
        format="JPEG",
        quality=95,
    )

    print("=" * 70)
    print("PATHOVERSE — TILE CONTACT SHEET")
    print("=" * 70)
    print("Tiles :", len(tiles))
    print("Rows  :", rows)
    print("Columns:", COLUMNS)
    print("Size  :", sheet.size)
    print()
    print("Saved:")
    print(OUTPUT_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
