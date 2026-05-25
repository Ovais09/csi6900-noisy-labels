import pandas as pd

from src.evaluation.experiment_runner import run_single_experiment


def test_run_single_experiment_returns_four_conditions():
    results = run_single_experiment(
        dataset_name="breast_cancer",
        noise_type="symmetric",
        noise_rate=0.1,
        model_name="logistic_regression",
        random_state=42,
        cv_folds=3,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 4
    assert set(results["condition"]) == {
        "clean_upper_bound",
        "noisy_baseline",
        "remove_suspicious",
        "relabel_suspicious",
    }
    assert "detector" in results.columns
    assert set(results["detector"]) == {"none", "combined"}
    assert results["accuracy"].notna().all()
    assert results["f1_macro"].notna().all()

    cleaning_rows = results[
        results["condition"].isin(["remove_suspicious", "relabel_suspicious"])
    ]
    assert cleaning_rows["det_precision"].notna().all()
    assert cleaning_rows["det_recall"].notna().all()
    assert cleaning_rows["det_f1"].notna().all()


def test_run_single_experiment_can_compare_detector_methods():
    results = run_single_experiment(
        dataset_name="breast_cancer",
        noise_type="symmetric",
        noise_rate=0.1,
        model_name="logistic_regression",
        random_state=42,
        cv_folds=3,
        detector_methods=("combined", "confidence"),
    )

    assert len(results) == 6
    assert set(results["detector"]) == {"none", "combined", "confidence"}

    cleaning_rows = results[
        results["condition"].isin(["remove_suspicious", "relabel_suspicious"])
    ]
    assert set(cleaning_rows["detector"]) == {"combined", "confidence"}
