"""Run clean and noisy baseline experiments without cleaning methods."""

import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_datasets import load_dataset
from src.data.preprocessing import make_train_val_test_split
from src.evaluation.metrics import classification_metrics
from src.models.baselines import get_baseline_models
from src.noise.inject_noise import inject_class_dependent_noise, inject_symmetric_noise
from src.utils.config import load_experiment_config
from src.utils.io import save_dataframe
from src.utils.seeds import set_global_seed


def _inject_noise(y_train, noise_type, noise_rate, random_state):
    if noise_type == "symmetric":
        return inject_symmetric_noise(y_train, noise_rate, random_state)
    if noise_type == "class_dependent":
        return inject_class_dependent_noise(y_train, noise_rate, random_state)
    raise ValueError(f"Unknown noise type: {noise_type}")


def _quick_config(config):
    quick = dict(config)
    quick["datasets"] = config["datasets"][:1]
    quick["noise_types"] = config["noise_types"][:1]
    quick["noise_rates"] = config["noise_rates"][:1]
    quick["models"] = config["models"][:1]
    quick["seeds"] = config["seeds"][:1]
    return quick


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--output", default="results/tables/baseline_results.csv")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.quick:
        config = _quick_config(config)

    output_path = args.output
    if args.quick and args.output == "results/tables/baseline_results.csv":
        output_path = "results/tables/baseline_results_quick.csv"

    rows = []
    combinations = product(
        config["datasets"],
        config["noise_types"],
        config["noise_rates"],
        config["models"],
        config["seeds"],
    )

    for dataset_name, noise_type, noise_rate, model_name, seed in combinations:
        set_global_seed(int(seed))
        print(
            "Running baseline "
            f"dataset={dataset_name}, "
            f"noise_type={noise_type}, "
            f"noise_rate={noise_rate}, "
            f"model={model_name}, "
            f"seed={seed}"
        )

        dataset = load_dataset(dataset_name)
        X_train, _, X_test, y_train, _, y_test = make_train_val_test_split(
            dataset.X,
            dataset.y,
            random_state=int(seed),
        )
        y_train_noisy, corrupted_mask = _inject_noise(
            y_train,
            noise_type=noise_type,
            noise_rate=float(noise_rate),
            random_state=int(seed),
        )
        model = get_baseline_models(random_state=int(seed))[model_name]

        for condition, labels in (
            ("clean_upper_bound", y_train),
            ("noisy_baseline", y_train_noisy),
        ):
            fitted_model = clone(model)
            fitted_model.fit(X_train, labels)
            metrics = classification_metrics(fitted_model, X_test, y_test)
            rows.append(
                {
                    "dataset": dataset_name,
                    "noise_type": noise_type,
                    "noise_rate": float(noise_rate),
                    "model": model_name,
                    "condition": condition,
                    "random_state": int(seed),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "n_corrupted": int(corrupted_mask.sum()),
                    **metrics,
                }
            )

    saved_path = save_dataframe(pd.DataFrame(rows), output_path)
    print(f"Saved baseline results to {saved_path}")


if __name__ == "__main__":
    main()
