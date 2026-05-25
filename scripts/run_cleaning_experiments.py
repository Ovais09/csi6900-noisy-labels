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
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    if args.quick:
        config = _quick_config(config)

    output_path = args.output or config["output"]["results_csv"]
    cv_folds = int(config.get("cv_folds", 5))
    detector_methods = config.get("detectors", ["combined"])
    all_results = []

    combinations = product(
        config["datasets"],
        config["noise_types"],
        config["noise_rates"],
        config["models"],
        config["seeds"],
    )

    for dataset, noise_type, noise_rate, model, seed in combinations:
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
        all_results.append(result)

    final_results = pd.concat(all_results, ignore_index=True)
    saved_path = save_dataframe(final_results, output_path)
    print(f"Saved results to {saved_path}")


if __name__ == "__main__":
    main()
