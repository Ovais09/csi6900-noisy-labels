"""Synthetic label-noise injection methods."""

import numpy as np
import pandas as pd


SUPPORTED_NOISE_TYPES = ("symmetric", "class_dependent")


def _validate_noise_rate(noise_rate: float) -> None:
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError(f"noise_rate must be between 0 and 1, got {noise_rate}")


def _as_series(y_array, index, name=None) -> pd.Series:
    return pd.Series(y_array, index=index, name=name)


def inject_symmetric_noise(y, noise_rate: float, random_state: int = 42):
    """Randomly flip a fraction of labels to any incorrect class.

    Returns
    -------
    y_noisy:
        Labels after corruption, with the original index preserved.
    corrupted_mask:
        Positional boolean mask where True means the label was corrupted.
    """

    _validate_noise_rate(noise_rate)
    rng = np.random.default_rng(random_state)

    y_array = np.asarray(y).copy()
    classes = np.unique(y_array)
    if len(classes) < 2:
        raise ValueError("At least two classes are required to inject label noise.")

    n_samples = len(y_array)
    n_corrupt = int(round(noise_rate * n_samples))
    corrupted_mask = np.zeros(n_samples, dtype=bool)

    if n_corrupt == 0:
        return _as_series(y_array, y.index, getattr(y, "name", None)), corrupted_mask

    corrupt_indices = rng.choice(n_samples, size=n_corrupt, replace=False)
    corrupted_mask[corrupt_indices] = True

    for idx in corrupt_indices:
        current_label = y_array[idx]
        possible_labels = classes[classes != current_label]
        y_array[idx] = rng.choice(possible_labels)

    return _as_series(y_array, y.index, getattr(y, "name", None)), corrupted_mask


def inject_class_dependent_noise(y, noise_rate: float, random_state: int = 42):
    """Flip labels using a fixed class-dependent cyclic mapping.

    For sorted classes ``[0, 1, 2]``, the mapping is ``0 -> 1``, ``1 -> 2``,
    and ``2 -> 0``. Class membership is computed from the original clean labels
    so a sample can be corrupted at most once.
    """

    _validate_noise_rate(noise_rate)
    rng = np.random.default_rng(random_state)

    clean_labels = np.asarray(y).copy()
    y_noisy = clean_labels.copy()
    classes = np.unique(clean_labels)
    if len(classes) < 2:
        raise ValueError("At least two classes are required to inject label noise.")

    class_to_next = {
        classes[i]: classes[(i + 1) % len(classes)] for i in range(len(classes))
    }

    corrupted_mask = np.zeros(len(clean_labels), dtype=bool)

    for cls in classes:
        cls_indices = np.where(clean_labels == cls)[0]
        n_corrupt_cls = int(round(noise_rate * len(cls_indices)))

        if n_corrupt_cls == 0:
            continue

        corrupt_indices = rng.choice(cls_indices, size=n_corrupt_cls, replace=False)
        y_noisy[corrupt_indices] = class_to_next[cls]
        corrupted_mask[corrupt_indices] = True

    return _as_series(y_noisy, y.index, getattr(y, "name", None)), corrupted_mask
