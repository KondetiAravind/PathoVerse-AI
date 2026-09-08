import argparse
import json

from pathoverse.datasets.downloader import download_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download a PathoVerse dataset."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "pathmnist",
            "patchcamelyon",
        ],
    )

    args = parser.parse_args()

    result = download_dataset(args.dataset)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()