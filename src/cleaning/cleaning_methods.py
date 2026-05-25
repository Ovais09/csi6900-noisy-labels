"""Cleaning actions applied after suspicious-label detection."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _validate_fraction(fraction: float) -> None:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")


def _make_cv(y_train, cv: int, random_state: int) -> StratifiedKFold:
    y_array = np.asarray(y_train)
    _, counts = np.unique(y_array, return_counts=True)
    max_splits = int(counts.min())
    n_splits = min(cv, max_splits)

    if n_splits < 2:
        raise ValueError("At least two samples per class are required for CV relabeling.")

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def select_top_k_suspicious(scores, fraction: float):
    """Select the top fraction of suspiciousness scores."""

    _validate_fraction(fraction)
    scores = np.asarray(scores)
    k = int(round(fraction * len(scores)))
    suspicious_mask = np.zeros(len(scores), dtype=bool)

    if k <= 0:
        return suspicious_mask

    top_indices = np.argsort(scores, kind="mergesort")[-k:]
    suspicious_mask[top_indices] = True

    return suspicious_mask


def remove_suspicious_samples(X_train, y_train, suspicious_mask):
    """Drop rows flagged as suspicious from the training set."""

    keep_mask = ~np.asarray(suspicious_mask, dtype=bool)
    return X_train.iloc[keep_mask], y_train.iloc[keep_mask]


def relabel_suspicious_samples(
    model,
    X_train,
    y_train,
    suspicious_mask,
    cv: int = 5,
    random_state: int = 42,
):
    """Replace suspicious labels with cross-validated model predictions."""

    suspicious_mask = np.asarray(suspicious_mask, dtype=bool)
    y_pred = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=_make_cv(y_train, cv=cv, random_state=random_state),
        method="predict",
        n_jobs=1,
    )

    y_relabel = y_train.copy()
    y_relabel.iloc[suspicious_mask] = y_pred[suspicious_mask]

    return y_relabel
