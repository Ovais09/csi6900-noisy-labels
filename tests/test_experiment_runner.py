import pandas as pd
import pytest

from src.evaluation.experiment_runner import (
    run_single_experiment,
    run_single_experiment_details,
)


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


def test_experiment_is_exactly_reproducible_for_fixed_seed():
    kwargs = {
        "dataset_name": "wine",
        "noise_type": "class_dependent",
        "noise_rate": 0.2,
        "model_name": "logistic_regression",
        "random_state": 123,
        "cv_folds": 3,
        "detector_methods": ("confidence", "combined"),
    }

    first = run_single_experiment(**kwargs)
    second = run_single_experiment(**kwargs)

    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_experiment_details_use_the_same_results_and_models():
    kwargs = {
        "dataset_name": "breast_cancer",
        "noise_type": "symmetric",
        "noise_rate": 0.1,
        "model_name": "logistic_regression",
        "random_state": 42,
        "cv_folds": 3,
        "detector_methods": ("confidence",),
    }

    details = run_single_experiment_details(**kwargs)
    results = run_single_experiment(**kwargs)

    pd.testing.assert_frame_equal(details["results"], results, check_exact=True)
    assert set(details["models"]) == {
        ("clean_upper_bound", "none"),
        ("noisy_baseline", "none"),
        ("remove_suspicious", "confidence"),
        ("relabel_suspicious", "confidence"),
    }
    assert len(details["X_validation"]) == len(details["y_validation"])


def test_experiment_details_can_release_fitted_models():
    details = run_single_experiment_details(
        dataset_name="breast_cancer",
        noise_type="symmetric",
        noise_rate=0.1,
        model_name="logistic_regression",
        random_state=42,
        cv_folds=3,
        detector_methods=("confidence",),
        retain_models=False,
    )

    assert details["models"] == {}
    assert len(details["results"]) == 4


@pytest.mark.parametrize(
    "detectors",
    [(), ("confidence", "confidence"), ("unknown",)],
)
def test_experiment_rejects_invalid_detector_lists(detectors):
    with pytest.raises(ValueError):
        run_single_experiment(
            dataset_name="breast_cancer",
            noise_type="symmetric",
            noise_rate=0.1,
            model_name="logistic_regression",
            random_state=42,
            cv_folds=3,
            detector_methods=detectors,
        )
