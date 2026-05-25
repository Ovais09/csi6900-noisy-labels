"""Summarize experiment results across random seeds."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import save_dataframe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/tables/main_experiment_results.csv",
        help="Input experiment CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/tables/summary_results.csv",
        help="Output summary CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    if "detector" not in df.columns:
        df["detector"] = "combined"

    group_cols = [
        "dataset",
        "noise_type",
        "noise_rate",
        "model",
        "condition",
        "detector",
    ]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            det_precision_mean=("det_precision", "mean"),
            det_recall_mean=("det_recall", "mean"),
            det_f1_mean=("det_f1", "mean"),
            n_corrupted_mean=("n_corrupted", "mean"),
            n_suspicious_mean=("n_suspicious", "mean"),
        )
        .reset_index()
    )

    saved_path = save_dataframe(summary, args.output)
    print(f"Saved summary to {saved_path}")


if __name__ == "__main__":
    main()
