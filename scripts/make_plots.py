"""Create figures from experiment result CSVs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
            return row["condition"]
        return f"{row['condition']} ({row['detector']})"

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

    plt.figure(figsize=(8, 5))
    if sns is not None:
        _seaborn_lineplot(
            data=subset,
            x="noise_rate",
            y=metric,
            hue="plot_condition",
            marker="o",
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
    plt.title(f"{metric} vs Noise Rate: {dataset}, {noise_type}, {model}")
    plt.xlabel("Noise Rate")
    plt.ylabel(metric)
    plt.tight_layout()

    filename = f"{dataset}_{noise_type}_{model}_{metric}.png"
    path = output_dir / filename
    plt.savefig(path, dpi=300)
    plt.close()
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

    plt.figure(figsize=(8, 5))
    if sns is not None:
        _seaborn_lineplot(
            data=subset,
            x="noise_rate",
            y="det_f1",
            hue="detector",
            style="noise_type",
            marker="o",
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
    plt.title(f"Detection F1 vs Noise Rate: {dataset}, {model}")
    plt.xlabel("Noise Rate")
    plt.ylabel("Detection F1")
    plt.tight_layout()

    path = output_dir / f"{dataset}_{model}_detection_f1.png"
    plt.savefig(path, dpi=300)
    plt.close()
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
    df = add_plot_labels(pd.read_csv(args.input))
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
