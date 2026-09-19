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
from src.evaluation.result_contract import COMBO_COLUMNS, combo_key, complete_combo_keys
from src.utils.config import load_experiment_config, quick_experiment_config
from src.utils.io import save_dataframe, save_dataframe_atomic
from src.utils.seeds import set_global_seed


def _quick_config(config):
    return quick_experiment_config(config)


def _result_output_path(config, requested_path=None, quick: bool = False):
    if requested_path:
        return requested_path
    configured_path = Path(config["output"]["results_csv"])
    if quick:
        return str(
            configured_path.with_name(
                f"{configured_path.stem}_quick{configured_path.suffix}"
            )
        )
    return str(configured_path)


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
    return combo_key(dataset, noise_type, noise_rate, model, seed)


def _completed_keys(output_path, detector_methods):
    path = Path(output_path)
    if not path.exists():
        return set()

    df = pd.read_csv(path)
    return complete_combo_keys(df, detector_methods)


def _prepare_resume_file(output_path, detector_methods, allowed_combo_keys=None):
    """Drop incomplete combinations before safely resuming a result file."""

    path = Path(output_path)
    if not path.exists():
        return set()

    existing = pd.read_csv(path)
    completed = complete_combo_keys(existing, detector_methods)
    if allowed_combo_keys is not None:
        completed.intersection_update(allowed_combo_keys)
    row_keys = existing.loc[:, list(COMBO_COLUMNS)].apply(
        lambda row: combo_key(*row),
        axis=1,
    )
    retained = existing.loc[row_keys.isin(completed)].copy()

    removed_rows = len(existing) - len(retained)
    if removed_rows:
        print(
            f"Removed {removed_rows} incomplete or out-of-config result rows "
            "before resuming "
            f"{output_path}"
        )
        save_dataframe_atomic(retained, path)

    return completed


def _append_results(result, output_path, replace: bool = False):
    """Atomically append one complete experiment combination."""

    path = Path(output_path)
    if path.exists() and not replace:
        existing = pd.read_csv(path)
        combined = (
            result.reset_index(drop=True)
            if existing.empty
            else pd.concat([existing, result], ignore_index=True)
        )
    else:
        combined = result.reset_index(drop=True)
    save_dataframe_atomic(combined, path)


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.quick:
        config = _quick_config(config)

    output_path = _result_output_path(
        config,
        requested_path=args.output,
        quick=args.quick,
    )
    cv_folds = int(config.get("cv_folds", 5))
    detector_methods = config.get("detectors", ["combined"])
    output_file = Path(output_path)

    combinations = list(
        product(
            config["datasets"],
            config["noise_types"],
            config["noise_rates"],
            config["models"],
            config["seeds"],
        )
    )
    expected_keys = {_combo_key(*combination) for combination in combinations}

    completed_keys = (
        _prepare_resume_file(
            output_path,
            detector_methods,
            allowed_combo_keys=expected_keys,
        )
        if args.resume
        else set()
    )
    replace_existing = not args.resume

    for dataset, noise_type, noise_rate, model, seed in combinations:
        current_key = _combo_key(dataset, noise_type, noise_rate, model, seed)
        if current_key in completed_keys:
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
        _append_results(result, output_path, replace=replace_existing)
        replace_existing = False

    if not output_file.exists():
        empty_results = pd.DataFrame()
        saved_path = save_dataframe(empty_results, output_path)
    else:
        saved_path = output_file
    print(f"Saved results to {saved_path}")


if __name__ == "__main__":
    main()
