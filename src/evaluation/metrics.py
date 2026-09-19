"""Classification and noisy-label detection metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(model, X_test, y_test):
    """Evaluate a fitted classifier on clean test labels."""

    y_pred = model.predict(X_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "auroc": np.nan,
    }

    if not hasattr(model, "predict_proba"):
        return results

    y_proba = model.predict_proba(X_test)

    try:
        if y_proba.shape[1] == 2:
            results["auroc"] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            results["auroc"] = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
    except ValueError:
        results["auroc"] = np.nan

    return results


def detection_metrics(true_corrupted_mask, predicted_suspicious_mask):
    """Evaluate how well a detector recovered the intentionally corrupted rows."""

    true_corrupted_mask = np.asarray(true_corrupted_mask)
    predicted_suspicious_mask = np.asarray(predicted_suspicious_mask)
    if (
        true_corrupted_mask.ndim != 1
        or predicted_suspicious_mask.ndim != 1
        or len(true_corrupted_mask) != len(predicted_suspicious_mask)
    ):
        raise ValueError("Detection masks must be one-dimensional and equally sized")

    return {
        "det_precision": precision_score(
            true_corrupted_mask,
            predicted_suspicious_mask,
            zero_division=0,
        ),
        "det_recall": recall_score(
            true_corrupted_mask,
            predicted_suspicious_mask,
            zero_division=0,
        ),
        "det_f1": f1_score(
            true_corrupted_mask,
            predicted_suspicious_mask,
            zero_division=0,
        ),
    }
