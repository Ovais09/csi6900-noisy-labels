"""Create figures from experiment result CSVs."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_dataframe
from src.utils.plotting import CONDITION_LABELS, save_report_figure

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


def _seaborn_lineplot(**kwargs):
    try:
        sns.lineplot(**kwargs, errorbar="sd")
    except TypeError:
        sns.lineplot(**kwargs, ci="sd")


def add_plot_labels(df):
    df = df.copy()

    def label_row(row):
        if row["condition"] in {"clean_upper_bound", "noisy_baseline"}:
            return CONDITION_LABELS[row["condition"]]
        return f"{CONDITION_LABELS[row['condition']]} ({row['detector']})"

    if "detector" not in df.columns:
        df["detector"] = "combined"

    df["plot_condition"] = df.apply(label_row, axis=1)
    return df


def plot_metric(df, metric, dataset, noise_type, model, output_dir):
    subset = df[
        (df["dataset"] == dataset)
        & (df["noise_type"] == noise_type)
        & (df["model"] == model)
    ]

    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if sns is not None:
        _seaborn_lineplot(
            data=subset,
            x="noise_rate",
            y=metric,
            hue="plot_condition",
            style="plot_condition",
            markers=True,
            ax=ax,
        )
    else:
        for condition, group in subset.groupby("plot_condition"):
            grouped = group.groupby("noise_rate")[metric].mean().reset_index()
            plt.plot(
                grouped["noise_rate"],
                grouped[metric],
                marker="o",
                label=condition,
            )
        plt.legend()
    ax.set_title(f"{dataset.replace('_', ' ').title()} | {model.replace('_', ' ')}\n"
                 f"{noise_type.replace('_', ' ')} noise; mean +/- 1 seed SD")
    ax.set_xlabel("Training label noise")
    ax.set_ylabel({"accuracy": "Accuracy", "f1_macro": "Macro F1"}.get(metric, metric))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(sorted(subset["noise_rate"].unique()))
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title=None, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()

    filename = f"{dataset}_{noise_type}_{model}_{metric}.png"
    path = output_dir / filename
    save_report_figure(fig, path)
    plt.close(fig)
    print(f"Saved {path}")


def detection_plot_data(df, dataset, model):
    """Return one detector observation per experiment, without cleaning duplicates."""

    return df[
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (df["condition"] == "remove_suspicious")
    ]


def plot_detection_f1(df, dataset, model, output_dir):
    subset = detection_plot_data(df, dataset, model)

    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if sns is not None:
        _seaborn_lineplot(
            data=subset,
            x="noise_rate",
            y="det_f1",
            hue="detector",
            style="noise_type",
            markers=True,
            ax=ax,
        )
    else:
        for (detector, noise_type), group in subset.groupby(["detector", "noise_type"]):
            grouped = group.groupby("noise_rate")["det_f1"].mean().reset_index()
            plt.plot(
                grouped["noise_rate"],
                grouped["det_f1"],
                marker="o",
                label=f"{detector}, {noise_type}",
            )
        plt.legend()
    ax.set_title(f"{dataset.replace('_', ' ').title()} | {model.replace('_', ' ')}\n"
                 "Detection F1; mean +/- 1 seed SD")
    ax.set_xlabel("Training label noise")
    ax.set_ylabel("Detection F1")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks(sorted(subset["noise_rate"].unique()))
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()

    path = output_dir / f"{dataset}_{model}_detection_f1.png"
    save_report_figure(fig, path)
    plt.close(fig)
    print(f"Saved {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/tables/main_experiment_results.csv",
        help="Input experiment CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = add_plot_labels(load_dataframe(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset]
        for model in sorted(dataset_df["model"].unique()):
            plot_detection_f1(df, dataset, model, output_dir)

        for noise_type in sorted(dataset_df["noise_type"].unique()):
            noise_df = dataset_df[dataset_df["noise_type"] == noise_type]
            for model in sorted(noise_df["model"].unique()):
                plot_metric(df, "accuracy", dataset, noise_type, model, output_dir)
                plot_metric(df, "f1_macro", dataset, noise_type, model, output_dir)


if __name__ == "__main__":
    main()
