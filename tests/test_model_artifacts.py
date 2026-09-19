from pathlib import Path

import joblib
import pandas as pd
import pytest

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


def test_create_model_diagnostics_writes_tables_and_figures(tmp_path):
    table_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"

    result = create_model_diagnostics(
        dataset_name="breast_cancer",
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
    assert result["model_metadata"] is not None
    assert (table_dir / "diagnostic_summary.csv").exists()
    assert (table_dir / "classification_report_by_condition.csv").exists()
    assert len(list(figure_dir.glob("*confusion_matrix.png"))) == 4
    assert len(list(figure_dir.glob("*roc_curve.png"))) == 4
    assert len(list(figure_dir.glob("*precision_recall_curve.png"))) == 4
    assert (figure_dir / "macro_f1_by_condition.png").exists()
