from copy import deepcopy

import pytest

from scripts.run_cleaning_experiments import _result_output_path
from src.utils.config import load_experiment_config, validate_experiment_config


def _valid_config():
    return {
        "datasets": ["breast_cancer"],
        "noise_types": ["symmetric"],
        "noise_rates": [0.1],
        "models": ["logistic_regression"],
        "detectors": ["confidence"],
        "seeds": [42],
        "cv_folds": 3,
        "output": {
            "results_csv": "results/tables/custom.csv",
            "summary_csv": "results/tables/custom_summary.csv",
        },
    }


def test_repository_experiment_config_is_valid():
    config = load_experiment_config("configs/experiment_config.yaml")

    assert "dry_bean" in config["datasets"]
    assert config["cv_folds"] == 5


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("datasets", ["unknown"], "unsupported"),
        ("detectors", ["confidence", "confidence"], "duplicate"),
        ("noise_rates", [1.0], r"\[0, 1\)"),
        ("seeds", [True], "integers"),
        ("cv_folds", 1, "at least 2"),
    ],
)
def test_invalid_config_values_fail_early(key, value, message):
    config = deepcopy(_valid_config())
    config[key] = value

    with pytest.raises(ValueError, match=message):
        validate_experiment_config(config)


def test_config_requires_output_paths():
    config = _valid_config()
    del config["output"]["summary_csv"]

    with pytest.raises(ValueError, match="summary_csv"):
        validate_experiment_config(config)


def test_quick_output_does_not_overwrite_full_results():
    config = _valid_config()

    assert _result_output_path(config) == "results/tables/custom.csv"
    assert _result_output_path(config, quick=True) == "results/tables/custom_quick.csv"
    assert (
        _result_output_path(config, requested_path="elsewhere.csv", quick=True)
        == "elsewhere.csv"
    )
