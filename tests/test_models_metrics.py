import numpy as np

from src.data.load_datasets import load_dataset
from src.data.preprocessing import make_train_val_test_split
from src.evaluation.metrics import classification_metrics, detection_metrics
from src.models.baselines import get_baseline_models


def test_baseline_model_factory_exposes_expected_models():
    models = get_baseline_models(random_state=42)

    assert set(models) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }


def test_classification_metrics_include_expected_keys():
    dataset = load_dataset("wine")
    X_train, _, X_test, y_train, _, y_test = make_train_val_test_split(
        dataset.X,
        dataset.y,
        random_state=42,
    )
    model = get_baseline_models(random_state=42)["logistic_regression"]

    model.fit(X_train, y_train)
    metrics = classification_metrics(model, X_test, y_test)

    assert set(metrics) == {"accuracy", "f1_macro", "auroc"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0


def test_detection_metrics_handles_empty_predictions():
    true_mask = np.array([True, False, True, False])
    predicted_mask = np.array([False, False, False, False])

    metrics = detection_metrics(true_mask, predicted_mask)

    assert metrics == {
        "det_precision": 0.0,
        "det_recall": 0.0,
        "det_f1": 0.0,
    }


def test_detection_metrics_rejects_mismatched_masks():
    with np.testing.assert_raises(ValueError):
        detection_metrics([True, False], [True])
