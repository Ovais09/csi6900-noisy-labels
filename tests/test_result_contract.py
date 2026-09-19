import pandas as pd
import pytest

from scripts.run_cleaning_experiments import (
    _append_results,
    _completed_keys,
    _prepare_resume_file,
)
from scripts.validate_results import validate_results, validation_report
from src.evaluation.experiment_runner import run_single_experiment
from src.evaluation.result_contract import combo_key, complete_combo_keys


DETECTORS = ("combined", "confidence")


def _results():
    return run_single_experiment(
        dataset_name="breast_cancer",
        noise_type="symmetric",
        noise_rate=0.1,
        model_name="logistic_regression",
        random_state=42,
        cv_folds=3,
        detector_methods=DETECTORS,
    )


def _config():
    return {
        "datasets": ["breast_cancer"],
        "noise_types": ["symmetric"],
        "noise_rates": [0.1],
        "models": ["logistic_regression"],
        "detectors": list(DETECTORS),
        "seeds": [42],
    }


def test_complete_combo_keys_requires_exact_row_contract():
    results = _results()
    expected_key = combo_key(
        "breast_cancer", "symmetric", 0.1, "logistic_regression", 42
    )

    assert complete_combo_keys(results, DETECTORS) == {expected_key}
    assert complete_combo_keys(results.iloc[:-1], DETECTORS) == set()

    wrong_detector = results.copy()
    wrong_detector.loc[
        wrong_detector["condition"] == "remove_suspicious", "detector"
    ] = "none"
    assert complete_combo_keys(wrong_detector, DETECTORS) == set()


def test_resume_removes_partial_combo_before_rerun(tmp_path):
    output_path = tmp_path / "results.csv"
    partial = _results().iloc[:-1]
    partial.to_csv(output_path, index=False)

    completed = _prepare_resume_file(output_path, DETECTORS)

    assert completed == set()
    assert pd.read_csv(output_path).empty

    _append_results(_results(), output_path)
    assert len(pd.read_csv(output_path)) == len(_results())
    assert len(_completed_keys(output_path, DETECTORS)) == 1


def test_resume_rejects_result_file_without_contract_columns(tmp_path):
    output_path = tmp_path / "bad.csv"
    pd.DataFrame({"dataset": ["breast_cancer"]}).to_csv(output_path, index=False)

    with pytest.raises(ValueError, match="missing resume columns"):
        _prepare_resume_file(output_path, DETECTORS)


def test_validator_rejects_wrong_condition_detector_contract():
    malformed = _results().copy()
    cleaning_index = malformed.index[
        malformed["condition"].eq("remove_suspicious")
        & malformed["detector"].eq("combined")
    ][0]
    malformed.loc[cleaning_index, "detector"] = "none"

    report = validation_report(validate_results(malformed, _config()))
    failures = set(report.loc[report["status"] == "FAIL", "check"])

    assert "row_contract" in failures
    assert "detectors" not in failures


def test_validator_rejects_missing_detection_metrics():
    malformed = _results().copy()
    malformed.loc[
        malformed["condition"].eq("remove_suspicious"), "det_f1"
    ] = float("nan")

    report = validation_report(validate_results(malformed, _config()))
    failures = set(report.loc[report["status"] == "FAIL", "check"])

    assert "det_f1_presence" in failures
    assert "detector_metric_consistency" in failures
