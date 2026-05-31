"""Run the full noisy-label cleaning experiment matrix."""

import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.experiment_runner import run_single_experiment
from src.utils.io import load_yaml, save_dataframe
from src.utils.seeds import set_global_seed


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
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output CSV path.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the first dataset/noise/rate/model/seed combination.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip combinations already present in the output CSV.",
    )
    return parser.parse_args()


def _combo_key(dataset, noise_type, noise_rate, model, seed):
    return (dataset, noise_type, float(noise_rate), model, int(seed))


def _completed_keys(output_path):
    path = Path(output_path)
    if not path.exists():
        return set()

    df = pd.read_csv(path)
    required_cols = {"dataset", "noise_type", "noise_rate", "model", "random_state"}
    if not required_cols.issubset(df.columns):
        return set()

    return {
        _combo_key(row.dataset, row.noise_type, row.noise_rate, row.model, row.random_state)
        for row in df[list(required_cols)].drop_duplicates().itertuples(index=False)
    }


def _append_results(result, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    args = parse_args()
    config = load_yaml(args.config)
    if args.quick:
        config = _quick_config(config)

    output_path = args.output or config["output"]["results_csv"]
    cv_folds = int(config.get("cv_folds", 5))
    detector_methods = config.get("detectors", ["combined"])
    output_file = Path(output_path)
    if output_file.exists() and not args.resume:
        output_file.unlink()

    completed_keys = _completed_keys(output_path) if args.resume else set()

    combinations = product(
        config["datasets"],
        config["noise_types"],
        config["noise_rates"],
        config["models"],
        config["seeds"],
    )

    for dataset, noise_type, noise_rate, model, seed in combinations:
        combo_key = _combo_key(dataset, noise_type, noise_rate, model, seed)
        if combo_key in completed_keys:
            print(
                "Skipping completed "
                f"dataset={dataset}, "
                f"noise_type={noise_type}, "
                f"noise_rate={noise_rate}, "
                f"model={model}, "
                f"seed={seed}"
            )
            continue

        set_global_seed(int(seed))
        print(
            "Running "
            f"dataset={dataset}, "
            f"noise_type={noise_type}, "
            f"noise_rate={noise_rate}, "
            f"model={model}, "
            f"detectors={','.join(detector_methods)}, "
            f"seed={seed}"
        )

        result = run_single_experiment(
            dataset_name=dataset,
            noise_type=noise_type,
            noise_rate=float(noise_rate),
            model_name=model,
            random_state=int(seed),
            cv_folds=cv_folds,
            detector_methods=detector_methods,
        )
        _append_results(result, output_path)

    if not output_file.exists():
        empty_results = pd.DataFrame()
        saved_path = save_dataframe(empty_results, output_path)
    else:
        saved_path = output_file
    print(f"Saved results to {saved_path}")


if __name__ == "__main__":
    main()
