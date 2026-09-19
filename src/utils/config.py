"""Validation and loading for experiment configuration files."""

import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

from src.cleaning.detectors import SUPPORTED_DETECTORS
from src.data.load_datasets import SUPPORTED_DATASETS
from src.models.baselines import SUPPORTED_MODEL_NAMES
from src.noise.inject_noise import SUPPORTED_NOISE_TYPES
from src.utils.io import load_yaml


MATRIX_KEYS = (
    "datasets",
    "noise_types",
    "noise_rates",
    "models",
    "detectors",
    "seeds",
)
QUICK_MATRIX_KEYS = ("datasets", "noise_types", "noise_rates", "models", "seeds")


def _validate_unique_sequence(config, key, allowed=None):
    values = config.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"config.{key} must be a non-empty list")
    try:
        unique_values = set(values)
    except TypeError as error:
        raise ValueError(f"config.{key} values must be scalar") from error
    if len(values) != len(unique_values):
        raise ValueError(f"config.{key} contains duplicate values")
    if allowed is not None:
        unknown = sorted(unique_values.difference(allowed), key=str)
        if unknown:
            raise ValueError(f"config.{key} contains unsupported values: {unknown}")
    return values


def validate_experiment_config(config):
    """Validate and return an experiment configuration mapping."""

    if not isinstance(config, Mapping):
        raise ValueError("Experiment config must be a mapping")

    _validate_unique_sequence(config, "datasets", SUPPORTED_DATASETS)
    _validate_unique_sequence(config, "noise_types", SUPPORTED_NOISE_TYPES)
    noise_rates = _validate_unique_sequence(config, "noise_rates")
    _validate_unique_sequence(config, "models", SUPPORTED_MODEL_NAMES)
    _validate_unique_sequence(config, "detectors", SUPPORTED_DETECTORS)
    seeds = _validate_unique_sequence(config, "seeds")

    for rate in noise_rates:
        if isinstance(rate, bool) or not isinstance(rate, Real):
            raise ValueError("config.noise_rates must contain only numbers")
        if not math.isfinite(float(rate)) or not 0.0 <= float(rate) < 1.0:
            raise ValueError("config.noise_rates values must be in [0, 1)")

    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("config.seeds must contain only integers")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("config.seeds values must be in [0, 2**32 - 1]")

    cv_folds = config.get("cv_folds", 5)
    if isinstance(cv_folds, bool) or not isinstance(cv_folds, int) or cv_folds < 2:
        raise ValueError("config.cv_folds must be an integer of at least 2")

    output = config.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("config.output must be a mapping")
    for key in ("results_csv", "summary_csv"):
        value = output.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"config.output.{key} must be a non-empty path string")
    if Path(output["results_csv"]) == Path(output["summary_csv"]):
        raise ValueError("config output paths for results and summary must differ")

    return config


def load_experiment_config(path):
    """Load a YAML experiment config and validate its complete schema."""

    return validate_experiment_config(load_yaml(path))


def quick_experiment_config(config):
    """Return the one-combination smoke-test subset of a valid config."""

    quick = dict(config)
    for key in QUICK_MATRIX_KEYS:
        quick[key] = list(config[key][:1])
    quick["detectors"] = list(config["detectors"])
    quick["output"] = dict(config["output"])
    return quick
