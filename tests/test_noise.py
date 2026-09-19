import numpy as np
import pandas as pd
import pytest

from src.noise.inject_noise import (
    inject_class_dependent_noise,
    inject_symmetric_noise,
)


def test_symmetric_noise_corrupts_expected_fraction():
    y = pd.Series([0, 1] * 50)

    y_noisy, corrupted_mask = inject_symmetric_noise(
        y,
        noise_rate=0.2,
        random_state=42,
    )

    assert corrupted_mask.sum() == 20
    assert y_noisy.index.equals(y.index)
    assert np.all(y_noisy[corrupted_mask].to_numpy() != y[corrupted_mask].to_numpy())
    assert np.all(y_noisy[~corrupted_mask].to_numpy() == y[~corrupted_mask].to_numpy())


def test_class_dependent_noise_uses_original_class_membership():
    y = pd.Series([0] * 10 + [1] * 10 + [2] * 10)

    y_noisy, corrupted_mask = inject_class_dependent_noise(
        y,
        noise_rate=0.5,
        random_state=42,
    )

    assert corrupted_mask.sum() == 15
    assert set(y_noisy[corrupted_mask].unique()).issubset({0, 1, 2})
    assert np.all(y_noisy[corrupted_mask].to_numpy() != y[corrupted_mask].to_numpy())
    assert np.all(y_noisy[~corrupted_mask].to_numpy() == y[~corrupted_mask].to_numpy())


def test_noise_rate_validation():
    y = pd.Series([0, 1, 0, 1])

    with pytest.raises(ValueError):
        inject_symmetric_noise(y, -0.1)

    with pytest.raises(ValueError):
        inject_class_dependent_noise(y, 1.1)

    with pytest.raises(ValueError):
        inject_symmetric_noise(y, float("nan"))

    with pytest.raises(ValueError):
        inject_symmetric_noise(y, True)


def test_noise_injection_accepts_array_labels_with_range_index():
    y = np.array([0, 1] * 10)

    y_noisy, mask = inject_symmetric_noise(y, 0.25, random_state=7)

    assert isinstance(y_noisy, pd.Series)
    assert y_noisy.index.equals(pd.RangeIndex(len(y)))
    assert mask.sum() == 5


def test_noise_injection_rejects_non_vector_labels():
    y = np.array([[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="one-dimensional"):
        inject_symmetric_noise(y, 0.2)
