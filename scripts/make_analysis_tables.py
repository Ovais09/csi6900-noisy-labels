"""Create compact analysis tables from noisy-label experiment results."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import save_dataframe


BASE_COLUMNS = ["dataset", "noise_type", "noise_rate", "model", "random_state"]


def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Average classification metrics by dataset and training condition."""

    return (
        df.groupby(["dataset", "condition", "detector"], dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            rows=("accuracy", "size"),
        )
        .reset_index()
        .sort_values(["dataset", "accuracy_mean"], ascending=[True, False])
    )


def detector_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank detector variants by detection quality and downstream accuracy."""

    cleaning_df = df[df["condition"].isin(["remove_suspicious", "relabel_suspicious"])]
    return (
        cleaning_df.groupby(["detector"], dropna=False)
        .agg(
            det_precision_mean=("det_precision", "mean"),
            det_recall_mean=("det_recall", "mean"),
            det_f1_mean=("det_f1", "mean"),
            accuracy_mean=("accuracy", "mean"),
            f1_macro_mean=("f1_macro", "mean"),
            rows=("accuracy", "size"),
        )
        .reset_index()
        .sort_values(["det_f1_mean", "accuracy_mean"], ascending=False)
    )


def cleaning_gain_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Measure remove/relabel gains against the noisy baseline."""

    noisy = (
        df[df["condition"] == "noisy_baseline"]
        .set_index(BASE_COLUMNS)["accuracy"]
        .rename("noisy_accuracy")
    )
    cleaned = (
        df[df["condition"].isin(["remove_suspicious", "relabel_suspicious"])]
        .set_index(BASE_COLUMNS + ["condition", "detector"])["accuracy"]
        .rename("cleaned_accuracy")
        .reset_index()
    )

    gains = cleaned.join(noisy, on=BASE_COLUMNS)
    gains["accuracy_gain"] = gains["cleaned_accuracy"] - gains["noisy_accuracy"]

    return (
        gains.groupby(["condition", "detector"], dropna=False)
        .agg(
            mean_gain=("accuracy_gain", "mean"),
            median_gain=("accuracy_gain", "median"),
            pct_positive=("accuracy_gain", lambda values: (values > 0).mean()),
            worst_gain=("accuracy_gain", "min"),
            best_gain=("accuracy_gain", "max"),
            rows=("accuracy_gain", "size"),
        )
        .reset_index()
        .sort_values(["mean_gain", "pct_positive"], ascending=False)
    )


def noise_impact_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how much label noise hurts before cleaning."""

    clean = (
        df[df["condition"] == "clean_upper_bound"]
        .set_index(BASE_COLUMNS)["accuracy"]
        .rename("clean_accuracy")
    )
    noisy = (
        df[df["condition"] == "noisy_baseline"]
        .set_index(BASE_COLUMNS)["accuracy"]
        .rename("noisy_accuracy")
    )
    impact = pd.concat([clean, noisy], axis=1).reset_index()
    impact["accuracy_drop"] = impact["clean_accuracy"] - impact["noisy_accuracy"]

    return (
        impact.groupby(["dataset", "noise_type", "noise_rate", "model"], dropna=False)
        .agg(
            clean_accuracy_mean=("clean_accuracy", "mean"),
            noisy_accuracy_mean=("noisy_accuracy", "mean"),
            accuracy_drop_mean=("accuracy_drop", "mean"),
            accuracy_drop_std=("accuracy_drop", "std"),
            rows=("accuracy_drop", "size"),
        )
        .reset_index()
        .sort_values("accuracy_drop_mean", ascending=False)
    )


def best_cleaning_by_setting(df: pd.DataFrame) -> pd.DataFrame:
    """Find the best remove/relabel detector for each dataset/noise/model/rate setting."""

    cleaning_df = df[df["condition"].isin(["remove_suspicious", "relabel_suspicious"])]
    group_cols = ["dataset", "noise_type", "noise_rate", "model", "condition", "detector"]
    mean_scores = (
        cleaning_df.groupby(group_cols, dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            f1_macro_mean=("f1_macro", "mean"),
            det_f1_mean=("det_f1", "mean"),
        )
        .reset_index()
    )

    rank_cols = ["dataset", "noise_type", "noise_rate", "model"]
    order = mean_scores.sort_values(
        rank_cols + ["accuracy_mean", "f1_macro_mean", "det_f1_mean"],
        ascending=[True, True, True, True, False, False, False],
    )
    return order.groupby(rank_cols, as_index=False).head(1).reset_index(drop=True)


def make_all_analysis_tables(df: pd.DataFrame):
    """Return all report-oriented analysis tables."""

    return {
        "condition_summary": condition_summary(df),
        "detector_ranking": detector_ranking(df),
        "cleaning_gain_summary": cleaning_gain_summary(df),
        "noise_impact_summary": noise_impact_summary(df),
        "best_cleaning_by_setting": best_cleaning_by_setting(df),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/tables/main_experiment_results.csv",
        help="Input experiment CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables/analysis",
        help="Directory for generated analysis CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)

    for name, table in make_all_analysis_tables(df).items():
        path = output_dir / f"{name}.csv"
        saved_path = save_dataframe(table, path)
        print(f"Saved {saved_path}")


if __name__ == "__main__":
    main()
