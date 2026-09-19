"""Generate report-ready metadata tables for the experiment section."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_datasets import load_dataset
from src.utils.config import load_experiment_config, quick_experiment_config
from src.utils.io import save_dataframe

MODEL_ROWS = [
    {
        "model": "logistic_regression",
        "type": "linear classifier",
        "reason_included": "Simple interpretable baseline for tabular classification.",
    },
    {
        "model": "random_forest",
        "type": "bagged tree ensemble",
        "reason_included": "Common robust nonlinear baseline for tabular data.",
    },
    {
        "model": "hist_gradient_boosting",
        "type": "boosted tree ensemble",
        "reason_included": "Strong built-in scikit-learn tabular model.",
    },
]

NOISE_DESCRIPTIONS = {
    "symmetric": "Randomly flip selected training labels to any incorrect class.",
    "class_dependent": "Flip selected training labels to the next class cyclically.",
}

DETECTOR_ROWS = [
    {
        "detector": "combined",
        "description": "Weighted confidence, cross-validated disagreement, and margin score.",
    },
    {
        "detector": "confidence",
        "description": "Suspiciousness is one minus the probability of the observed label.",
    },
    {
        "detector": "disagreement",
        "description": "Suspiciousness is one when CV prediction disagrees with the label.",
    },
    {
        "detector": "margin",
        "description": "Suspiciousness is one minus the top-vs-second probability margin.",
    },
]

CLEANING_ROWS = [
    {
        "condition": "clean_upper_bound",
        "description": "Train on clean labels for a best-case reference.",
    },
    {
        "condition": "noisy_baseline",
        "description": "Train directly on corrupted training labels.",
    },
    {
        "condition": "remove_suspicious",
        "description": "Remove top-scoring suspicious training samples before retraining.",
    },
    {
        "condition": "relabel_suspicious",
        "description": "Replace suspicious labels with CV predictions before retraining.",
    },
]


def dataset_table(dataset_names):
    rows = []

    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name)
        rows.append(
            {
                "dataset": dataset.name,
                "samples": len(dataset.y),
                "features": dataset.X.shape[1],
                "classes": dataset.n_classes,
                "task_type": dataset.task_type,
            }
        )

    return pd.DataFrame(rows)


def noise_table(noise_types, noise_rates):
    return pd.DataFrame(
        [
            {
                "noise_type": noise_type,
                "description": NOISE_DESCRIPTIONS[noise_type],
                "rates": ", ".join(f"{int(rate * 100)}%" for rate in noise_rates),
            }
            for noise_type in noise_types
        ]
    )


def model_table(model_names):
    """Return metadata only for models selected by the experiment config."""

    selected = set(model_names)
    return pd.DataFrame([row for row in MODEL_ROWS if row["model"] in selected])


def detector_table(detector_names):
    """Return metadata only for detectors selected by the experiment config."""

    selected = set(detector_names)
    return pd.DataFrame(
        [row for row in DETECTOR_ROWS if row["detector"] in selected]
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.quick:
        config = quick_experiment_config(config)
    output_dir = Path(args.output_dir)

    tables = {
        "dataset_overview.csv": dataset_table(config["datasets"]),
        "model_overview.csv": model_table(config["models"]),
        "noise_settings.csv": noise_table(config["noise_types"], config["noise_rates"]),
        "detector_methods.csv": detector_table(config["detectors"]),
        "cleaning_conditions.csv": pd.DataFrame(CLEANING_ROWS),
    }

    for filename, table in tables.items():
        path = save_dataframe(table, output_dir / filename)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
