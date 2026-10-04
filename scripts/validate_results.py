"""Validate saved experiment results against the configured experiment matrix."""

import argparse
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.result_contract import (
    BASE_CONDITIONS,
    CLEANING_CONDITIONS,
    COMBO_COLUMNS,
    ROW_KEY_COLUMNS,
    combo_key,
    expected_row_signatures,
)
from src.utils.config import load_experiment_config
from src.utils.io import load_dataframe, save_dataframe

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


@dataclass(frozen=True)
class ValidationIssue:
    status: str
    check: str
    detail: str


def _combo_key(dataset, noise_type, noise_rate, model, seed):
    return combo_key(dataset, noise_type, noise_rate, model, seed)


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


def _rate_is_valid(series, allow_na: bool = False):
    if series.empty:
        return False
    numeric = pd.to_numeric(series, errors="coerce")
    if not allow_na and numeric.isna().any():
        return False
    valid = numeric.dropna().between(0.0, 1.0)
    return bool(valid.all())


def _integer_series_is_valid(series, allow_na: bool = False):
    numeric = pd.to_numeric(series, errors="coerce")
    if not allow_na and numeric.isna().any():
        return False
    values = numeric.dropna().to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and np.equal(values, np.floor(values)).all())


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

    combo_cols = list(COMBO_COLUMNS)
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

    row_key_cols = list(ROW_KEY_COLUMNS)
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

    expected_signatures = expected_row_signatures(detectors)
    bad_contracts = 0
    for _, group in df.groupby(combo_cols, sort=False, dropna=False):
        signatures = list(zip(group["condition"], group["detector"]))
        if len(signatures) != len(expected_signatures):
            bad_contracts += 1
            continue
        if len(signatures) != len(set(signatures)):
            bad_contracts += 1
            continue
        if set(signatures) != expected_signatures:
            bad_contracts += 1

    issues.append(
        _issue(
            "PASS" if bad_contracts == 0 else "FAIL",
            "row_contract",
            "Every combination has the exact condition/detector rows"
            if bad_contracts == 0
            else f"{bad_contracts} combinations violate the row contract",
        )
    )

    for metric in CLASSIFICATION_METRICS:
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

    expected_suspicious = (cleaning_rows["noise_rate"] * cleaning_rows["n_train"]).round()
    suspicious_count_ok = bool(
        _integer_series_is_valid(cleaning_rows["n_suspicious"])
        and (cleaning_rows["n_suspicious"] == expected_suspicious).all()
        and (cleaning_rows["n_suspicious"] >= 0).all()
        and (cleaning_rows["n_suspicious"] <= cleaning_rows["n_train"]).all()
    )
    issues.append(
        _issue(
            "PASS" if suspicious_count_ok else "FAIL",
            "n_suspicious_count",
            "Cleaning rows select the configured fraction of training rows",
        )
    )
    issues.append(
        _issue(
            "PASS" if suspicious_ok else "FAIL",
            "n_suspicious_presence",
            "n_suspicious is blank for baselines and present for cleaning rows",
        )
    )

    corrupted_ok = bool(
        _integer_series_is_valid(df["n_corrupted"])
        and (df["n_corrupted"] >= 0).all()
        and (df["n_corrupted"] <= df["n_train"]).all()
    )
    issues.append(
        _issue(
            "PASS" if corrupted_ok else "FAIL",
            "n_corrupted_presence",
            "n_corrupted is an integer between zero and n_train",
        )
    )

    train_test_ok = bool(
        _integer_series_is_valid(df["n_train"])
        and _integer_series_is_valid(df["n_test"])
        and (df["n_train"] > 0).all()
        and (df["n_test"] > 0).all()
    )
    issues.append(
        _issue(
            "PASS" if train_test_ok else "FAIL",
            "split_sizes",
            "n_train and n_test are positive integers",
        )
    )

    symmetric_rows = df[df["noise_type"] == "symmetric"]
    expected_symmetric_corruption = (
        symmetric_rows["noise_rate"] * symmetric_rows["n_train"]
    ).round()
    symmetric_corruption_ok = bool(
        (
            symmetric_rows["n_corrupted"] == expected_symmetric_corruption
        ).all()
    )
    issues.append(
        _issue(
            "PASS" if symmetric_corruption_ok else "FAIL",
            "symmetric_corruption_count",
            "Symmetric corruption count matches the configured noise fraction",
        )
    )

    count_columns = ["n_train", "n_test", "n_corrupted"]
    inconsistent_counts = (
        df.groupby(combo_cols, dropna=False)[count_columns].nunique(dropna=False) > 1
    ).any(axis=1)
    issues.append(
        _issue(
            "PASS" if not inconsistent_counts.any() else "FAIL",
            "combo_count_consistency",
            "Train, test, and corrupted counts agree within each combination",
        )
    )

    detector_group_cols = combo_cols + ["detector"]
    detector_consistency = (
        cleaning_rows.groupby(detector_group_cols, dropna=False)[
            ["n_suspicious", *DETECTION_METRICS]
        ].nunique(dropna=False)
        > 1
    ).any(axis=1)
    issues.append(
        _issue(
            "PASS" if not detector_consistency.any() else "FAIL",
            "detector_metric_consistency",
            "Remove and relabel rows share detector metrics",
        )
    )

    precision = pd.to_numeric(cleaning_rows["det_precision"], errors="coerce").to_numpy()
    recall = pd.to_numeric(cleaning_rows["det_recall"], errors="coerce").to_numpy()
    expected_f1 = np.divide(2 * precision * recall, precision + recall,
                            out=np.zeros_like(precision), where=(precision + recall) != 0)
    tp_precision = precision * cleaning_rows["n_suspicious"].to_numpy()
    tp_recall = recall * cleaning_rows["n_corrupted"].to_numpy()
    detector_arithmetic_ok = (
        np.allclose(expected_f1, cleaning_rows["det_f1"], rtol=0, atol=1e-10)
        and np.allclose(tp_precision, tp_recall, rtol=0, atol=1e-8)
        and np.allclose(tp_precision, np.round(tp_precision), rtol=0, atol=1e-8)
    )
    issues.append(_issue(
        "PASS" if detector_arithmetic_ok else "FAIL", "detection_arithmetic",
        "Precision, recall, F1, and integer true-positive counts agree",
    ))
    correct_predictions = pd.to_numeric(df["accuracy"], errors="coerce") * df["n_test"]
    accuracy_counts_ok = np.allclose(
        correct_predictions, np.round(correct_predictions), rtol=0, atol=1e-8
    )
    issues.append(_issue(
        "PASS" if accuracy_counts_ok else "FAIL", "accuracy_count_arithmetic",
        "Accuracy corresponds to an integer number of correct test predictions",
    ))

    clean_rows = df[df["condition"] == "clean_upper_bound"]
    clean_group_cols = ["dataset", "model", "random_state"]
    clean_invariance = (
        clean_rows.groupby(clean_group_cols, dropna=False)[
            ["accuracy", "f1_macro", "auroc", "n_train", "n_test"]
        ].nunique(dropna=False)
        > 1
    ).any(axis=1)
    issues.append(
        _issue(
            "PASS" if not clean_invariance.any() else "FAIL",
            "clean_baseline_invariance",
            "Clean baselines are invariant to noise type and rate",
        )
    )

    numeric_metrics = df[list(CLASSIFICATION_METRICS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite_ok = bool(np.isfinite(numeric_metrics.to_numpy(dtype=float)).all())
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
    df = load_dataframe(args.results)
    config = load_experiment_config(args.config)

    issues = validate_results(df, config)
    report = validation_report(issues)
    saved_path = save_dataframe(report, args.output)

    print(report.to_string(index=False))
    print(f"Saved validation report to {saved_path}")

    if (report["status"] == "FAIL").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
