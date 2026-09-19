"""Noisy-label suspiciousness scoring methods."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict


SUPPORTED_DETECTORS = ("combined", "confidence", "disagreement", "margin")


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


def cross_validated_probabilities(model, X_train, y_train, cv: int, random_state: int):
    """Return cross-validated class probabilities for each training row."""

    return cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=_make_cv(y_train, cv=cv, random_state=random_state),
        method="predict_proba",
        n_jobs=1,
    )


def predicted_labels_from_probabilities(proba, y_train):
    """Convert probability columns into labels using the observed class order."""

    classes = np.unique(np.asarray(y_train))
    return classes[np.argmax(proba, axis=1)]


def scores_from_probabilities(
    proba,
    y_train,
    detector: str = "combined",
    confidence_weight: float = 0.5,
    disagreement_weight: float = 0.3,
    margin_weight: float = 0.2,
):
    """Compute a detector score from already-computed CV probabilities."""

    normalized_detector = detector.lower()
    proba = np.asarray(proba, dtype=float)
    y_array = np.asarray(y_train)
    classes = np.unique(y_array)
    if proba.ndim != 2:
        raise ValueError("proba must be a two-dimensional array")
    if proba.shape != (len(y_array), len(classes)):
        raise ValueError(
            "proba shape must equal (number of labels, number of classes)"
        )
    if not np.isfinite(proba).all() or np.any(proba < 0.0) or np.any(proba > 1.0):
        raise ValueError("proba must contain finite probabilities in [0, 1]")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("each probability row must sum to 1")
    class_to_col = {cls: i for i, cls in enumerate(classes)}

    assigned_label_probs = np.array(
        [proba[i, class_to_col[label]] for i, label in enumerate(y_array)]
    )
    confidence = 1.0 - assigned_label_probs

    predicted_labels = predicted_labels_from_probabilities(proba, y_train)
    disagreement = (predicted_labels != y_array).astype(float)

    sorted_probs = np.sort(proba, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    margin_suspiciousness = 1.0 - margin

    if normalized_detector == "combined":
        return (
            confidence_weight * confidence
            + disagreement_weight * disagreement
            + margin_weight * margin_suspiciousness
        )

    if normalized_detector == "confidence":
        return confidence

    if normalized_detector == "disagreement":
        return disagreement

    if normalized_detector == "margin":
        return margin_suspiciousness

    raise ValueError(
        f"Unknown detector: {detector}. "
        f"Supported detectors are: {', '.join(SUPPORTED_DETECTORS)}"
    )


def confidence_scores(model, X_train, y_train, cv: int = 5, random_state: int = 42):
    """Score labels by low cross-validated probability of the observed label.

    Higher scores are more suspicious.
    """

    y_array = np.asarray(y_train)
    proba = cross_validated_probabilities(
        model, X_train, y_train, cv=cv, random_state=random_state
    )

    return scores_from_probabilities(proba, y_array, detector="confidence")


def disagreement_scores(model, X_train, y_train, cv: int = 5, random_state: int = 42):
    """Score labels by cross-validated prediction disagreement."""

    y_pred = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=_make_cv(y_train, cv=cv, random_state=random_state),
        method="predict",
        n_jobs=1,
    )
    y_array = np.asarray(y_train)

    return (y_pred != y_array).astype(float)


def margin_scores(model, X_train, y_train, cv: int = 5, random_state: int = 42):
    """Score labels by low cross-validated prediction margin."""

    proba = cross_validated_probabilities(
        model, X_train, y_train, cv=cv, random_state=random_state
    )

    return scores_from_probabilities(proba, y_train, detector="margin")


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

    proba = cross_validated_probabilities(
        model, X_train, y_train, cv=cv, random_state=random_state
    )

    return scores_from_probabilities(
        proba,
        y_train,
        detector="combined",
        confidence_weight=confidence_weight,
        disagreement_weight=disagreement_weight,
        margin_weight=margin_weight,
    )


def suspicious_scores(
    model,
    X_train,
    y_train,
    detector: str = "combined",
    cv: int = 5,
    random_state: int = 42,
):
    """Return suspiciousness scores for a named detector.

    Supported detectors are:

    - ``combined``: confidence + cross-validated disagreement + margin.
    - ``confidence``: low probability assigned to the observed label.
    - ``disagreement``: cross-validated predicted label differs from observed label.
    - ``margin``: low top-vs-second probability margin.
    """

    normalized_detector = detector.lower()

    if normalized_detector == "combined":
        return combined_suspicious_scores(
            model,
            X_train,
            y_train,
            cv=cv,
            random_state=random_state,
        )

    if normalized_detector == "confidence":
        return confidence_scores(
            model,
            X_train,
            y_train,
            cv=cv,
            random_state=random_state,
        )

    if normalized_detector == "disagreement":
        return disagreement_scores(
            model,
            X_train,
            y_train,
            cv=cv,
            random_state=random_state,
        )

    if normalized_detector == "margin":
        return margin_scores(
            model,
            X_train,
            y_train,
            cv=cv,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown detector: {detector}. "
        f"Supported detectors are: {', '.join(SUPPORTED_DETECTORS)}"
    )
