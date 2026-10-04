from pathlib import Path

import joblib
import pandas as pd
import pytest

from scripts.audit_release import _check_diagnostics, _check_model_artifacts
from scripts.make_model_diagnostics import create_model_diagnostics
from scripts.train_final_models import (
    MODEL_CONDITIONS,
    save_final_model_set,
    train_final_model_set,
)


def test_train_final_model_set_returns_expected_models_and_metrics():
    artifacts = train_final_model_set(
        dataset_name="breast_cancer",
        model_name="logistic_regression",
        noise_type="symmetric",
        noise_rate=0.1,
        detector="confidence",
        random_state=42,
        cv_folds=3,
    )

    assert set(artifacts["models"]) == set(MODEL_CONDITIONS)
    assert set(artifacts["metrics"]["condition"]) == set(MODEL_CONDITIONS)
    assert artifacts["metrics"]["accuracy"].between(0.0, 1.0).all()
    assert artifacts["suspicious_mask"].sum() == artifacts["metrics"]["n_suspicious"].dropna().iloc[0]


def test_save_final_model_set_writes_reloadable_joblib_models(tmp_path):
    artifacts = train_final_model_set(
        dataset_name="breast_cancer",
        model_name="logistic_regression",
        noise_type="symmetric",
        noise_rate=0.1,
        detector="confidence",
        random_state=42,
        cv_folds=3,
    )
    metadata = save_final_model_set(artifacts, tmp_path)

    assert set(metadata["condition"]) == set(MODEL_CONDITIONS)
    for model_path in metadata["model_path"]:
        path = Path(model_path)
        assert path.exists()
        assert hasattr(joblib.load(path), "predict")
    assert metadata["model_sha256"].str.len().eq(64).all()
    assert metadata["model_bytes"].gt(0).all()
    assert not list(tmp_path.glob("*.tmp"))


def test_model_save_failure_preserves_existing_artifact(tmp_path, monkeypatch):
    existing_path = tmp_path / "clean_upper_bound.joblib"
    existing_path.write_bytes(b"previous-valid-artifact")
    artifacts = {
        "metrics": pd.DataFrame({"condition": ["clean_upper_bound"]}),
        "models": {"clean_upper_bound": object()},
    }

    def fail_after_partial_write(_model, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(joblib, "dump", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        save_final_model_set(artifacts, tmp_path)

    assert existing_path.read_bytes() == b"previous-valid-artifact"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("dataset_name", ["breast_cancer", "wine"])
def test_create_model_diagnostics_writes_tables_and_figures(tmp_path, dataset_name):
    table_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"

    result = create_model_diagnostics(
        dataset_name=dataset_name,
        model_name="logistic_regression",
        noise_type="symmetric",
        noise_rate=0.1,
        detector="confidence",
        random_state=42,
        cv_folds=3,
        table_dir=table_dir,
        figure_dir=figure_dir,
        save_models=True,
        model_output_dir=tmp_path / "models",
    )

    assert not result["summary"].empty
    assert not result["classification_reports"].empty
    reports = result["classification_reports"]
    assert "accuracy" not in set(reports["label"])
    assert reports["support"].mod(1).eq(0).all()
    assert result["model_metadata"] is not None
    assert (table_dir / "diagnostic_summary.csv").exists()
    assert (table_dir / "classification_report_by_condition.csv").exists()
    assert len(list(figure_dir.glob("*confusion_matrix.png"))) == 4
    assert len(list(figure_dir.glob("*roc_curve.png"))) == 4
    assert len(list(figure_dir.glob("*precision_recall_curve.png"))) == 4
    assert (figure_dir / "macro_f1_by_condition.png").exists()

    metadata = result["model_metadata"]
    _check_diagnostics(table_dir, figure_dir, metadata)
    model_args = {
        "dataset_name": dataset_name, "model_name": "logistic_regression",
        "noise_type": "symmetric", "noise_rate": 0.1, "detector": "confidence",
        "random_state": 42, "model_output_dir": tmp_path / "models", "project_root": tmp_path,
        "results": result["summary"].drop(columns=["cv_folds", "n_validation",
                                                  "diagnostic_roc_auc", "diagnostic_pr_auc"]),
        "cv_folds": 3,
    }
    _check_model_artifacts(**model_args)
    model_args["results"].loc[0, "accuracy"] = 0.0
    with pytest.raises(ValueError, match="canonical result matrix"):
        _check_model_artifacts(**model_args)

    report_path = table_dir / "classification_report_by_condition.csv"
    report = pd.read_csv(report_path)
    stale_report = report.copy()
    stale_report.loc[0, "support"] += 1
    stale_report.to_csv(report_path, index=False)
    with pytest.raises(ValueError, match="fresh recomputation"):
        _check_diagnostics(table_dir, figure_dir, metadata)
    report.to_csv(report_path, index=False)

    summary_path = table_dir / "diagnostic_summary.csv"
    stale_summary = pd.read_csv(summary_path)
    stale_summary.loc[0, "diagnostic_pr_auc"] = 0.123
    stale_summary.to_csv(summary_path, index=False)
    with pytest.raises(ValueError, match="Recomputed diagnostic differs"):
        _check_diagnostics(table_dir, figure_dir, metadata)
