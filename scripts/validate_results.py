"""Validate saved experiment results against the configured experiment matrix."""

import argparse
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_yaml, save_dataframe


REQUIRED_COLUMNS = {
    "dataset",
    "noise_type",
    "noise_rate",
    "model",
    "condition",
    "detector",
    "random_state",
    "n_train",
    "n_test",
    "n_corrupted",
    "n_suspicious",
    "accuracy",
    "f1_macro",
    "auroc",
    "det_precision",
    "det_recall",
    "det_f1",
}

CLASSIFICATION_METRICS = ("accuracy", "f1_macro", "auroc")
DETECTION_METRICS = ("det_precision", "det_recall", "det_f1")
BASE_CONDITIONS = ("clean_upper_bound", "noisy_baseline")
CLEANING_CONDITIONS = ("remove_suspicious", "relabel_suspicious")


@dataclass(frozen=True)
class ValidationIssue:
    status: str
    check: str
    detail: str


def _combo_key(dataset, noise_type, noise_rate, model, seed):
    return (dataset, noise_type, float(noise_rate), model, int(seed))


def expected_combo_keys(config):
    """Return all expected dataset/noise/rate/model/seed combinations."""

    return {
        _combo_key(dataset, noise_type, noise_rate, model, seed)
        for dataset, noise_type, noise_rate, model, seed in product(
            config["datasets"],
            config["noise_types"],
            config["noise_rates"],
            config["models"],
            config["seeds"],
        )
    }


def _issue(status, check, detail):
    return ValidationIssue(status=status, check=check, detail=detail)


def _rate_is_valid(series):
    valid = series.dropna().between(0.0, 1.0)
    return bool(valid.all())


def validate_results(df: pd.DataFrame, config: dict):
    """Validate result shape, coverage, duplicate rows, and metric ranges."""

    issues = []
    detectors = tuple(config.get("detectors", ("combined",)))
    expected_rows_per_combo = len(BASE_CONDITIONS) + len(CLEANING_CONDITIONS) * len(
        detectors
    )
    expected_keys = expected_combo_keys(config)

    missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        issues.append(
            _issue("FAIL", "required_columns", f"Missing columns: {missing_columns}")
        )
        return issues

    issues.append(
        _issue("PASS", "required_columns", f"Found {len(REQUIRED_COLUMNS)} columns")
    )

    combo_cols = ["dataset", "noise_type", "noise_rate", "model", "random_state"]
    actual_keys = {
        _combo_key(row.dataset, row.noise_type, row.noise_rate, row.model, row.random_state)
        for row in df[combo_cols].drop_duplicates().itertuples(index=False)
    }

    missing_keys = expected_keys.difference(actual_keys)
    unexpected_keys = actual_keys.difference(expected_keys)
    if missing_keys or unexpected_keys:
        detail = (
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}, "
            f"expected={len(expected_keys)}, actual={len(actual_keys)}"
        )
        issues.append(_issue("FAIL", "matrix_coverage", detail))
    else:
        issues.append(
            _issue("PASS", "matrix_coverage", f"All {len(expected_keys)} combinations")
        )

    expected_rows = len(expected_keys) * expected_rows_per_combo
    if len(df) == expected_rows:
        issues.append(_issue("PASS", "row_count", f"{len(df)} rows"))
    else:
        issues.append(
            _issue("FAIL", "row_count", f"expected={expected_rows}, actual={len(df)}")
        )

    row_key_cols = combo_cols + ["condition", "detector"]
    duplicate_count = int(df.duplicated(row_key_cols).sum())
    if duplicate_count:
        issues.append(_issue("FAIL", "duplicate_rows", f"{duplicate_count} duplicates"))
    else:
        issues.append(_issue("PASS", "duplicate_rows", "No duplicate result rows"))

    expected_conditions = set(BASE_CONDITIONS).union(CLEANING_CONDITIONS)
    unexpected_conditions = sorted(set(df["condition"]).difference(expected_conditions))
    if unexpected_conditions:
        issues.append(
            _issue(
                "FAIL",
                "conditions",
                f"Unexpected conditions: {unexpected_conditions}",
            )
        )
    else:
        issues.append(_issue("PASS", "conditions", "All conditions are recognized"))

    expected_detectors = set(detectors).union({"none"})
    unexpected_detectors = sorted(set(df["detector"]).difference(expected_detectors))
    if unexpected_detectors:
        issues.append(
            _issue(
                "FAIL",
                "detectors",
                f"Unexpected detectors: {unexpected_detectors}",
            )
        )
    else:
        issues.append(_issue("PASS", "detectors", "All detectors are recognized"))

    grouped_sizes = df.groupby(combo_cols).size()
    bad_combo_sizes = grouped_sizes[grouped_sizes != expected_rows_per_combo]
    if bad_combo_sizes.empty:
        issues.append(
            _issue(
                "PASS",
                "rows_per_combo",
                f"{expected_rows_per_combo} rows per combination",
            )
        )
    else:
        issues.append(
            _issue(
                "FAIL",
                "rows_per_combo",
                f"{len(bad_combo_sizes)} combinations have unexpected row counts",
            )
        )

    for metric in CLASSIFICATION_METRICS:
        if metric == "auroc":
            valid = df[metric].dropna().between(0.0, 1.0).all()
        else:
            valid = _rate_is_valid(df[metric])
        status = "PASS" if valid else "FAIL"
        issues.append(_issue(status, f"{metric}_range", "Values are in [0, 1]"))

    cleaning_rows = df[df["condition"].isin(CLEANING_CONDITIONS)]
    base_rows = df[df["condition"].isin(BASE_CONDITIONS)]

    for metric in DETECTION_METRICS:
        base_ok = bool(base_rows[metric].isna().all())
        cleaning_ok = _rate_is_valid(cleaning_rows[metric])
        if base_ok and cleaning_ok:
            issues.append(_issue("PASS", f"{metric}_presence", "Detection metric layout OK"))
        else:
            issues.append(
                _issue(
                    "FAIL",
                    f"{metric}_presence",
                    f"base_nan={base_ok}, cleaning_range_ok={cleaning_ok}",
                )
            )

    suspicious_ok = bool(base_rows["n_suspicious"].isna().all()) and bool(
        cleaning_rows["n_suspicious"].notna().all()
    )
    issues.append(
        _issue(
            "PASS" if suspicious_ok else "FAIL",
            "n_suspicious_presence",
            "n_suspicious is blank for baselines and present for cleaning rows",
        )
    )

    corrupted_ok = bool(df["n_corrupted"].notna().all()) and bool(
        (df["n_corrupted"] >= 0).all()
    )
    issues.append(
        _issue(
            "PASS" if corrupted_ok else "FAIL",
            "n_corrupted_presence",
            "n_corrupted is present and non-negative",
        )
    )

    train_test_ok = bool((df["n_train"] > 0).all() and (df["n_test"] > 0).all())
    issues.append(
        _issue(
            "PASS" if train_test_ok else "FAIL",
            "split_sizes",
            "n_train and n_test are positive",
        )
    )

    finite_required = ["accuracy", "f1_macro"]
    finite_ok = bool(
        df[finite_required]
        .apply(lambda column: column.map(math.isfinite))
        .all()
        .all()
    )
    issues.append(
        _issue(
            "PASS" if finite_ok else "FAIL",
            "finite_metrics",
            "Required classification metrics are finite",
        )
    )

    return issues


def validation_report(issues):
    """Convert validation issues into a DataFrame."""

    return pd.DataFrame(
        [{"status": issue.status, "check": issue.check, "detail": issue.detail} for issue in issues]
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="results/tables/main_experiment_results.csv",
        help="Experiment result CSV to validate.",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="Experiment YAML config.",
    )
    parser.add_argument(
        "--output",
        default="results/tables/validation_report.csv",
        help="CSV path for the validation report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.results)
    config = load_yaml(args.config)

    issues = validate_results(df, config)
    report = validation_report(issues)
    saved_path = save_dataframe(report, args.output)

    print(report.to_string(index=False))
    print(f"Saved validation report to {saved_path}")

    if (report["status"] == "FAIL").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
