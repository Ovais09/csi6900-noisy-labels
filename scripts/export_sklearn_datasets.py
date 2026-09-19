"""Export supported datasets to local processed CSV files."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_datasets import load_dataset
from src.utils.io import save_dataframe_atomic
from src.utils.provenance import sha256_file

DEFAULT_DATASETS = ("breast_cancer", "wine", "digits")
EXTERNAL_DATASETS = ("dry_bean",)


def export_dataset(dataset_name: str, output_dir) -> dict:
    """Export one supported sklearn dataset and return manifest metadata."""

    dataset = load_dataset(dataset_name)
    output_path = Path(output_dir) / f"{dataset.name}.csv"
    df = dataset.X.copy()
    df["target"] = dataset.y

    saved_path = save_dataframe_atomic(df, output_path)
    return {
        "dataset": dataset.name,
        "path": str(saved_path),
        "samples": len(dataset.y),
        "features": dataset.X.shape[1],
        "classes": dataset.n_classes,
        "task_type": dataset.task_type,
        "sha256": sha256_file(saved_path),
    }


def export_datasets(dataset_names, output_dir="datasets/processed") -> pd.DataFrame:
    """Export datasets and write a manifest CSV next to them."""

    manifest_rows = [
        export_dataset(dataset_name, output_dir=output_dir)
        for dataset_name in dataset_names
    ]
    manifest = pd.DataFrame(manifest_rows)
    save_dataframe_atomic(manifest, Path(output_dir) / "manifest.csv")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset names to export.",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Also export external datasets such as UCI Dry Bean.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/processed",
        help="Directory where CSV files will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_names = list(args.datasets)
    if args.include_external:
        dataset_names.extend(EXTERNAL_DATASETS)

    manifest = export_datasets(dataset_names, output_dir=args.output_dir)

    for row in manifest.itertuples(index=False):
        print(
            f"Saved {row.dataset}: {row.samples} rows, "
            f"{row.features} features, {row.classes} classes -> {row.path}"
        )
    print(f"Saved manifest -> {Path(args.output_dir) / 'manifest.csv'}")


if __name__ == "__main__":
    main()
