import argparse
import json

from pathoverse.datasets.validator import validate_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Validate a PathoVerse dataset."
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

    result = validate_dataset(args.dataset)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()