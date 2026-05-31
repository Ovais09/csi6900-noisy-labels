import pandas as pd

from scripts.make_analysis_tables import make_all_analysis_tables
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
        "cleaning_gain_summary",
        "noise_impact_summary",
        "best_cleaning_by_setting",
    }
    assert all(not table.empty for table in tables.values())
    assert set(tables["detector_ranking"]["detector"]) == {"combined", "confidence"}


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
