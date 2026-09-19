import pytest

from src.evaluation.experiment_runner import run_single_experiment


def test_pinned_reference_experiment_metrics_do_not_drift():
    results = run_single_experiment(
        dataset_name="breast_cancer",
        noise_type="symmetric",
        noise_rate=0.1,
        model_name="logistic_regression",
        random_state=42,
        cv_folds=5,
        detector_methods=("combined", "confidence", "disagreement", "margin"),
    )
    indexed = results.set_index(["condition", "detector"])

    assert len(results) == 10
    assert indexed.loc[("clean_upper_bound", "none"), "accuracy"] == pytest.approx(
        0.9912280701754386,
        abs=1e-12,
    )
    assert indexed.loc[("noisy_baseline", "none"), "accuracy"] == pytest.approx(
        0.9385964912280702,
        abs=1e-12,
    )
    assert indexed.loc[("remove_suspicious", "confidence"), "accuracy"] == pytest.approx(
        0.9736842105263158,
        abs=1e-12,
    )
    assert indexed.loc[("remove_suspicious", "confidence"), "det_f1"] == pytest.approx(
        0.8235294117647058,
        abs=1e-12,
    )
    assert indexed.loc[("remove_suspicious", "margin"), "det_f1"] == pytest.approx(
        0.17647058823529413,
        abs=1e-12,
    )
