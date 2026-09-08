import argparse
import json

from pathoverse.datasets.statistics import (
    compute_dataset_statistics,
    save_statistics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "pathmnist",
            "patchcamelyon",
        ],
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    statistics = compute_dataset_statistics(
        args.dataset
    )

    print(json.dumps(statistics, indent=2))

    if args.output:
        output_path = save_statistics(
            args.dataset,
            args.output,
        )

        print()
        print(f"Saved statistics to: {output_path}")


if __name__ == "__main__":
    main()