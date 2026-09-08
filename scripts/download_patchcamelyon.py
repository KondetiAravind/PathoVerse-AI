from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen, Request


BASE_URL = (
    "https://zenodo.org/record/2546921/files/"
)

FILES = {
    "camelyonpatch_level_2_split_train_x.h5.gz":
        "1571f514728f59376b705fc836ff4b63",

    "camelyonpatch_level_2_split_train_y.h5.gz":
        "35c2d7259d906cfc8143347bb8e05be7",

    "camelyonpatch_level_2_split_valid_x.h5.gz":
        "d8c2d60d490dbd479f8199bdfa0cf6ec",

    "camelyonpatch_level_2_split_valid_y.h5.gz":
        "60a7035772fbdb7f34eb86d4420cf66a",

    "camelyonpatch_level_2_split_test_x.h5.gz":
        "d5c2d60d490dbd479f8199bdfa0cf6ec",

    "camelyonpatch_level_2_split_test_y.h5.gz":
        "2b85f58b927af9964a4c15b8f7e8f179",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def download(url: str, destination: Path):

    request = Request(
        url,
        headers={
            "User-Agent": "PathoVerse-AI/1.0"
        },
    )

    with urlopen(request) as response, open(
        destination,
        "wb",
    ) as output:

        total = response.headers.get(
            "Content-Length"
        )

        total = (
            int(total)
            if total
            else None
        )

        downloaded = 0

        while True:

            chunk = response.read(
                1024 * 1024
            )

            if not chunk:
                break

            output.write(chunk)
            downloaded += len(chunk)

            if total:
                percent = (
                    downloaded / total
                ) * 100

                print(
                    f"\r  {downloaded / 1e9:.2f} GB "
                    f"/ {total / 1e9:.2f} GB "
                    f"({percent:.1f}%)",
                    end="",
                    flush=True,
                )

    print()


def extract_gzip(
    gzip_path: Path,
    output_path: Path,
):

    print(
        f"  Extracting → {output_path.name}"
    )

    with gzip.open(
        gzip_path,
        "rb",
    ) as source, open(
        output_path,
        "wb",
    ) as target:

        shutil.copyfileobj(
            source,
            target,
            length=1024 * 1024,
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/raw/patchcamelyon"
        ),
    )

    parser.add_argument(
        "--keep-compressed",
        action="store_true",
    )

    args = parser.parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "PATHOVERSE — PATCHCAMELYON DOWNLOAD"
    )
    print("=" * 72)

    for filename, expected_md5 in FILES.items():

        compressed_path = (
            args.output / filename
        )

        output_filename = filename[:-3]

        output_path = (
            args.output / output_filename
        )

        print()
        print("-" * 72)
        print(filename)
        print("-" * 72)

        # --------------------------------------------------
        # Already extracted
        # --------------------------------------------------

        if output_path.exists():

            print(
                f"[SKIP] {output_path.name} already exists"
            )

            continue

        # --------------------------------------------------
        # Download compressed file
        # --------------------------------------------------

        if not compressed_path.exists():

            url = (
                BASE_URL
                + filename
            )

            print(
                f"Downloading:\n{url}"
            )

            download(
                url,
                compressed_path,
            )

            print(
                f"Downloaded: "
                f"{compressed_path.stat().st_size / 1e9:.2f} GB"
            )

        else:

            print(
                "[SKIP] Compressed file already exists"
            )

        # --------------------------------------------------
        # Verify checksum
        # --------------------------------------------------

        print("Checking MD5...")

        actual_md5 = md5(
            compressed_path
        )

        print(
            f"Expected: {expected_md5}"
        )

        print(
            f"Actual  : {actual_md5}"
        )

        if actual_md5 != expected_md5:

            raise RuntimeError(
                f"MD5 mismatch for {filename}"
            )

        print(
            "✓ MD5 verified"
        )

        # --------------------------------------------------
        # Extract
        # --------------------------------------------------

        extract_gzip(
            compressed_path,
            output_path,
        )

        print(
            f"✓ Extracted: {output_path.name}"
        )

        # --------------------------------------------------
        # Remove compressed file
        # --------------------------------------------------

        if not args.keep_compressed:

            compressed_path.unlink()

            print(
                "✓ Removed compressed archive"
            )

    print()
    print("=" * 72)
    print(
        "PATCHCAMELYON DOWNLOAD COMPLETE"
    )
    print("=" * 72)

    for filename in FILES:

        extracted = (
            args.output
            / filename[:-3]
        )

        print(
            f"{extracted.name:<55}"
            f"{'OK' if extracted.exists() else 'MISSING'}"
        )


if __name__ == "__main__":
    main()
