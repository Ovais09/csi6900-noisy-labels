"""Download and prepare external public datasets used in extended experiments."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.external_datasets import prepare_dry_bean_dataset

SUPPORTED_EXTERNAL_DATASETS = ("dry_bean",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_EXTERNAL_DATASETS),
        choices=SUPPORTED_EXTERNAL_DATASETS,
        help="External datasets to download and prepare.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild processed files even if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for dataset in args.datasets:
        if dataset == "dry_bean":
            path = prepare_dry_bean_dataset(force=args.force)
            print(f"Prepared dry_bean -> {path}")


if __name__ == "__main__":
    main()
