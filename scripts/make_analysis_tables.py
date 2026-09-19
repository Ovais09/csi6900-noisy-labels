"""Create compact analysis tables from noisy-label experiment results."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import save_dataframe


BASE_COLUMNS = ["dataset", "noise_type", "noise_rate", "model", "random_state"]
SETTING_COLUMNS = ["dataset", "noise_type", "noise_rate", "model"]


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
    detection_df = cleaning_df.drop_duplicates(BASE_COLUMNS + ["detector"])
    detection = (
        detection_df.groupby(["detector"], dropna=False)
        .agg(
            det_precision_mean=("det_precision", "mean"),
            det_recall_mean=("det_recall", "mean"),
            det_f1_mean=("det_f1", "mean"),
            detection_evaluations=("det_f1", "size"),
        )
        .reset_index()
    )
    downstream = (
        cleaning_df.groupby(["detector"], dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            f1_macro_mean=("f1_macro", "mean"),
            downstream_rows=("accuracy", "size"),
        )
        .reset_index()
    )
    return (
        detection.merge(downstream, on="detector", validate="one_to_one")
        .sort_values(["det_f1_mean", "accuracy_mean"], ascending=False)
    )


def detector_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize detection metrics without duplicate remove/relabel rows."""

    cleaning_df = df[df["condition"].isin(["remove_suspicious", "relabel_suspicious"])]
    detection_df = cleaning_df.drop_duplicates(BASE_COLUMNS + ["detector"])
    return (
        detection_df.groupby(["dataset", "noise_type", "detector"], dropna=False)
        .agg(
            det_precision_mean=("det_precision", "mean"),
            det_precision_std=("det_precision", "std"),
            det_recall_mean=("det_recall", "mean"),
            det_recall_std=("det_recall", "std"),
            det_f1_mean=("det_f1", "mean"),
            det_f1_std=("det_f1", "std"),
            evaluations=("det_f1", "size"),
        )
        .reset_index()
        .sort_values(
            ["dataset", "noise_type", "det_f1_mean"],
            ascending=[True, True, False],
        )
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


def _bootstrap_mean_interval(values, rng, n_resamples: int = 5000):
    values = np.asarray(values, dtype=float)
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples < 1
    ):
        raise ValueError("n_resamples must be a positive integer")
    if len(values) == 0:
        return np.nan, np.nan
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def _holm_adjust(p_values):
    """Return Holm family-wise adjusted p-values in original row order."""

    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(len(p_values), dtype=float)
    running_max = 0.0

    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(p_values) - rank) * p_values[index])
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = running_max

    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_cleaning_comparisons(
    df: pd.DataFrame,
    n_resamples: int = 5000,
    random_state: int = 2026,
) -> pd.DataFrame:
    """Compare cleaning conditions with noisy baselines using paired runs."""

    rng = np.random.default_rng(random_state)
    scopes = [("all", df)] + [
        (dataset, df[df["dataset"] == dataset])
        for dataset in sorted(df["dataset"].unique())
    ]
    rows = []

    for scope, scoped_df in scopes:
        noisy = scoped_df[scoped_df["condition"] == "noisy_baseline"]
        cleaned = scoped_df[
            scoped_df["condition"].isin(["remove_suspicious", "relabel_suspicious"])
        ]

        for metric in ("accuracy", "f1_macro", "auroc"):
            baseline_runs = noisy[BASE_COLUMNS + [metric]].rename(
                columns={metric: "baseline_value"}
            )
            for (condition, detector), group in cleaned.groupby(
                ["condition", "detector"], sort=True
            ):
                cleaned_runs = group[BASE_COLUMNS + [metric]].rename(
                    columns={metric: "cleaned_value"}
                )
                paired_runs = cleaned_runs.merge(
                    baseline_runs,
                    on=BASE_COLUMNS,
                    how="outer",
                    validate="one_to_one",
                    indicator=True,
                )
                unmatched = paired_runs["_merge"] != "both"
                if unmatched.any():
                    raise ValueError(
                        "Cleaning and baseline rows have different exact seed coverage."
                    )
                if paired_runs[["cleaned_value", "baseline_value"]].isna().any().any():
                    raise ValueError("Paired classification metrics must not be missing.")

                paired_runs["difference"] = (
                    paired_runs["cleaned_value"] - paired_runs["baseline_value"]
                )
                paired = (
                    paired_runs.groupby(SETTING_COLUMNS, dropna=False)
                    .agg(
                        baseline_value=("baseline_value", "mean"),
                        cleaned_value=("cleaned_value", "mean"),
                        difference=("difference", "mean"),
                        seeds=("random_state", "nunique"),
                    )
                    .reset_index()
                )
                differences = paired["difference"].to_numpy()
                if len(differences) == 0:
                    continue

                ci_low, ci_high = _bootstrap_mean_interval(
                    differences,
                    rng=rng,
                    n_resamples=n_resamples,
                )
                if np.allclose(differences, 0.0):
                    p_value = 1.0
                else:
                    p_value = float(wilcoxon(differences).pvalue)

                std = differences.std(ddof=1) if len(differences) > 1 else 0.0
                effect_dz = differences.mean() / std if std > 0 else 0.0
                tolerance = 1e-12
                rows.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "condition": condition,
                        "detector": detector,
                        "settings": len(differences),
                        "seeds_per_setting_min": paired["seeds"].min(),
                        "seeds_per_setting_max": paired["seeds"].max(),
                        "baseline_mean": paired["baseline_value"].mean(),
                        "cleaned_mean": paired["cleaned_value"].mean(),
                        "mean_gain": differences.mean(),
                        "median_gain": np.median(differences),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "win_rate": np.mean(differences > tolerance),
                        "tie_rate": np.mean(np.abs(differences) <= tolerance),
                        "loss_rate": np.mean(differences < -tolerance),
                        "paired_effect_dz": effect_dz,
                        "wilcoxon_p": p_value,
                    }
                )

    comparisons = pd.DataFrame(rows)
    comparisons["wilcoxon_p_holm"] = comparisons.groupby(
        ["scope", "metric"], group_keys=False
    )["wilcoxon_p"].transform(_holm_adjust)
    return comparisons.sort_values(
        ["scope", "metric", "mean_gain"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def make_all_analysis_tables(df: pd.DataFrame):
    """Return all report-oriented analysis tables."""

    return {
        "condition_summary": condition_summary(df),
        "detector_ranking": detector_ranking(df),
        "detector_breakdown": detector_breakdown(df),
        "cleaning_gain_summary": cleaning_gain_summary(df),
        "noise_impact_summary": noise_impact_summary(df),
        "best_cleaning_by_setting": best_cleaning_by_setting(df),
        "paired_cleaning_comparisons": paired_cleaning_comparisons(df),
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
