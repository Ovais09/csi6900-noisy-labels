"""Noisy-label suspiciousness scoring methods."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _make_cv(y_train, cv: int, random_state: int) -> StratifiedKFold:
    y_array = np.asarray(y_train)
    _, counts = np.unique(y_array, return_counts=True)
    max_splits = int(counts.min())
    n_splits = min(cv, max_splits)

    if n_splits < 2:
        raise ValueError("At least two samples per class are required for CV scoring.")

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def _predict_proba_cv(model, X_train, y_train, cv: int, random_state: int):
    return cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=_make_cv(y_train, cv=cv, random_state=random_state),
        method="predict_proba",
        n_jobs=-1,
    )


def confidence_scores(model, X_train, y_train, cv: int = 5, random_state: int = 42):
    """Score labels by low cross-validated probability of the observed label.

    Higher scores are more suspicious.
    """

    y_array = np.asarray(y_train)
    classes = np.unique(y_array)
    class_to_col = {cls: i for i, cls in enumerate(classes)}
    proba = _predict_proba_cv(model, X_train, y_train, cv=cv, random_state=random_state)

    assigned_label_probs = np.array(
        [proba[i, class_to_col[label]] for i, label in enumerate(y_array)]
    )

    return 1.0 - assigned_label_probs


def disagreement_scores(model, X_train, y_train, cv: int = 5, random_state: int = 42):
    """Score labels by cross-validated prediction disagreement."""

    y_pred = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=_make_cv(y_train, cv=cv, random_state=random_state),
        method="predict",
        n_jobs=-1,
    )
    y_array = np.asarray(y_train)

    return (y_pred != y_array).astype(float)


def margin_scores(model, X_train, y_train, cv: int = 5, random_state: int = 42):
    """Score labels by low cross-validated prediction margin."""

    proba = _predict_proba_cv(model, X_train, y_train, cv=cv, random_state=random_state)
    sorted_probs = np.sort(proba, axis=1)
    top = sorted_probs[:, -1]
    second = sorted_probs[:, -2]
    margin = top - second

    return 1.0 - margin


def combined_suspicious_scores(
    model,
    X_train,
    y_train,
    cv: int = 5,
    confidence_weight: float = 0.5,
    disagreement_weight: float = 0.3,
    margin_weight: float = 0.2,
    random_state: int = 42,
):
    """Combine confidence, disagreement, and margin suspiciousness scores."""

    y_array = np.asarray(y_train)
    classes = np.unique(y_array)
    class_to_col = {cls: i for i, cls in enumerate(classes)}
    proba = _predict_proba_cv(model, X_train, y_train, cv=cv, random_state=random_state)

    assigned_label_probs = np.array(
        [proba[i, class_to_col[label]] for i, label in enumerate(y_array)]
    )
    confidence = 1.0 - assigned_label_probs

    predicted_labels = classes[np.argmax(proba, axis=1)]
    disagreement = (predicted_labels != y_array).astype(float)

    sorted_probs = np.sort(proba, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    margin_suspiciousness = 1.0 - margin

    return (
        confidence_weight * confidence
        + disagreement_weight * disagreement
        + margin_weight * margin_suspiciousness
    )
