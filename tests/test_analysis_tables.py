import numpy as np
import pandas as pd
import pytest

from scripts.make_analysis_tables import (
    _holm_adjust,
    make_all_analysis_tables,
    paired_cleaning_comparisons,
)
from scripts.make_metadata_tables import detector_table, model_table
from scripts.make_plots import detection_plot_data
from scripts.run_full_pipeline import build_pipeline_commands, parse_args
from scripts.validate_results import validate_results, validation_report
from src.evaluation.experiment_runner import run_single_experiment


def _small_results():
    return run_single_experiment(
        dataset_name="breast_cancer",
        noise_type="symmetric",
        noise_rate=0.1,
        model_name="logistic_regression",
        random_state=42,
        cv_folds=3,
        detector_methods=("combined", "confidence"),
    )


def _small_config():
    return {
        "datasets": ["breast_cancer"],
        "noise_types": ["symmetric"],
        "noise_rates": [0.1],
        "models": ["logistic_regression"],
        "detectors": ["combined", "confidence"],
        "seeds": [42],
    }


def test_validate_results_accepts_complete_matrix():
    report = validation_report(validate_results(_small_results(), _small_config()))

    assert isinstance(report, pd.DataFrame)
    assert not (report["status"] == "FAIL").any()
    assert "matrix_coverage" in set(report["check"])
    assert "row_count" in set(report["check"])


def test_validate_results_rejects_incomplete_matrix():
    incomplete = _small_results().iloc[:-1].copy()
    report = validation_report(validate_results(incomplete, _small_config()))

    failing_checks = set(report.loc[report["status"] == "FAIL", "check"])

    assert "row_count" in failing_checks
    assert "rows_per_combo" in failing_checks


def test_make_all_analysis_tables_returns_expected_tables():
    tables = make_all_analysis_tables(_small_results())

    assert set(tables) == {
        "condition_summary",
        "detector_ranking",
        "detector_breakdown",
        "cleaning_gain_summary",
        "noise_impact_summary",
        "best_cleaning_by_setting",
        "paired_cleaning_comparisons",
    }
    assert all(not table.empty for table in tables.values())
    assert set(tables["detector_ranking"]["detector"]) == {"combined", "confidence"}
    assert set(tables["detector_ranking"]["detection_evaluations"]) == {1}


def test_paired_comparisons_have_complete_pairing_and_valid_rates():
    comparisons = paired_cleaning_comparisons(
        _small_results(),
        n_resamples=200,
        random_state=7,
    )

    assert set(comparisons["scope"]) == {"all", "breast_cancer"}
    assert set(comparisons["metric"]) == {"accuracy", "f1_macro", "auroc"}
    assert set(comparisons["settings"]) == {1}
    assert set(comparisons["seeds_per_setting_min"]) == {1}
    assert set(comparisons["seeds_per_setting_max"]) == {1}
    assert comparisons["wilcoxon_p"].between(0.0, 1.0).all()
    assert comparisons["wilcoxon_p_holm"].between(0.0, 1.0).all()
    assert np.allclose(
        comparisons[["win_rate", "tie_rate", "loss_rate"]].sum(axis=1),
        1.0,
    )


def test_paired_comparisons_require_exact_seed_identity():
    mismatched = _small_results()
    row_index = mismatched.index[
        mismatched["condition"].eq("remove_suspicious")
        & mismatched["detector"].eq("confidence")
    ][0]
    mismatched.loc[row_index, "random_state"] = 999

    with pytest.raises(ValueError, match="exact seed coverage"):
        paired_cleaning_comparisons(mismatched, n_resamples=20)


def test_paired_comparisons_reject_invalid_bootstrap_count():
    with pytest.raises(ValueError, match="positive integer"):
        paired_cleaning_comparisons(_small_results(), n_resamples=0)


def test_holm_adjustment_is_monotonic_and_not_smaller_than_raw_p_values():
    raw = np.array([0.01, 0.04, 0.03, 0.20])
    adjusted = _holm_adjust(raw)

    assert np.all(adjusted >= raw)
    sorted_adjusted = adjusted[np.argsort(raw)]
    assert np.all(np.diff(sorted_adjusted) >= 0.0)


def test_run_full_pipeline_quick_mode_skips_full_validation(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_full_pipeline.py",
            "--quick",
            "--results",
            "results/tables/quick.csv",
            "--figure-dir",
            "results/figures/quick",
        ],
    )
    args = parse_args()
    commands = build_pipeline_commands(args)

    flattened = [" ".join(command) for command in commands]

    assert any("run_cleaning_experiments.py" in command for command in flattened)
    assert any("--quick" in command for command in flattened)
    assert any("make_analysis_tables.py" in command for command in flattened)
    assert not any("validate_results.py" in command for command in flattened)


def test_run_full_pipeline_quick_defaults_are_isolated(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_full_pipeline.py", "--quick"])
    args = parse_args()
    commands = build_pipeline_commands(args)
    flattened = [" ".join(command) for command in commands]

    assert any(
        "--output results/tables/quick_pipeline_results.csv" in command
        for command in flattened
    )
    assert any(
        "--output results/tables/quick_pipeline_summary.csv" in command
        for command in flattened
    )
    assert any(
        "--output-dir results/tables/quick_pipeline_analysis" in command
        for command in flattened
    )
    assert any(
        "--output-dir results/figures/quick_pipeline" in command
        for command in flattened
    )
    assert any(
        "make_run_manifest.py" in command and "--quick" in command
        for command in flattened
    )
    assert any(
        "make_metadata_tables.py" in command
        and "--output-dir results/tables/quick_pipeline_metadata" in command
        and "--quick" in command
        for command in flattened
    )
    assert not any("download_external_datasets.py" in command for command in flattened)


def test_detection_plot_data_counts_each_detector_once():
    data = detection_plot_data(
        _small_results(),
        dataset="breast_cancer",
        model="logistic_regression",
    )

    assert len(data) == 2
    assert set(data["detector"]) == {"combined", "confidence"}
    assert set(data["condition"]) == {"remove_suspicious"}


def test_metadata_tables_follow_selected_config_methods():
    models = model_table(["random_forest"])
    detectors = detector_table(["confidence", "combined"])

    assert list(models["model"]) == ["random_forest"]
    assert set(detectors["detector"]) == {"confidence", "combined"}


def test_run_full_pipeline_can_include_model_artifacts(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_full_pipeline.py",
            "--quick",
            "--skip-training",
            "--save-final-models",
            "--artifact-dataset",
            "breast_cancer",
            "--artifact-model",
            "logistic_regression",
        ],
    )
    args = parse_args()
    commands = build_pipeline_commands(args)
    flattened = [" ".join(command) for command in commands]

    assert any("train_final_models.py" in command for command in flattened)
    assert any("--dataset breast_cancer" in command for command in flattened)
    assert any("--model logistic_regression" in command for command in flattened)


def test_run_full_pipeline_combines_diagnostics_and_model_saving(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_full_pipeline.py",
            "--quick",
            "--skip-training",
            "--save-final-models",
            "--make-model-diagnostics",
        ],
    )
    args = parse_args()
    commands = build_pipeline_commands(args)
    flattened = [" ".join(command) for command in commands]

    assert any("make_model_diagnostics.py" in command for command in flattened)
    assert any("--save-models" in command for command in flattened)
    assert not any("train_final_models.py" in command for command in flattened)
