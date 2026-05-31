import numpy as np
import pytest

from src.cleaning.cleaning_methods import (
    relabel_suspicious_samples,
    remove_suspicious_samples,
    select_top_k_suspicious,
)
from src.cleaning.detectors import (
    combined_suspicious_scores,
    confidence_scores,
    cross_validated_probabilities,
    disagreement_scores,
    margin_scores,
    scores_from_probabilities,
    suspicious_scores,
)
from src.data.load_datasets import load_dataset
from src.data.preprocessing import make_train_val_test_split
from src.models.baselines import get_baseline_models


def test_detector_scores_have_expected_shape_and_range():
    dataset = load_dataset("wine")
    X_train, _, _, y_train, _, _ = make_train_val_test_split(
        dataset.X,
        dataset.y,
        random_state=42,
    )
    model = get_baseline_models(random_state=42)["logistic_regression"]

    for scorer in (
        confidence_scores,
        disagreement_scores,
        margin_scores,
        combined_suspicious_scores,
    ):
        scores = scorer(model, X_train, y_train, cv=3, random_state=42)

        assert scores.shape == (len(y_train),)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


def test_select_top_k_suspicious_uses_requested_fraction():
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

    suspicious_mask = select_top_k_suspicious(scores, fraction=0.4)

    assert suspicious_mask.tolist() == [False, True, False, True, False]


def test_named_suspicious_score_dispatcher_matches_supported_methods():
    dataset = load_dataset("wine")
    X_train, _, _, y_train, _, _ = make_train_val_test_split(
        dataset.X,
        dataset.y,
        random_state=42,
    )
    model = get_baseline_models(random_state=42)["logistic_regression"]

    for detector in ("combined", "confidence", "disagreement", "margin"):
        scores = suspicious_scores(
            model,
            X_train,
            y_train,
            detector=detector,
            cv=3,
            random_state=42,
        )
        assert scores.shape == (len(y_train),)

    with pytest.raises(ValueError):
        suspicious_scores(model, X_train, y_train, detector="unknown", cv=3)


def test_scores_from_probabilities_match_standalone_probability_scorers():
    dataset = load_dataset("wine")
    X_train, _, _, y_train, _, _ = make_train_val_test_split(
        dataset.X,
        dataset.y,
        random_state=42,
    )
    model = get_baseline_models(random_state=42)["logistic_regression"]
    proba = cross_validated_probabilities(
        model,
        X_train,
        y_train,
        cv=3,
        random_state=42,
    )

    assert np.allclose(
        scores_from_probabilities(proba, y_train, detector="combined"),
        combined_suspicious_scores(model, X_train, y_train, cv=3, random_state=42),
    )
    assert np.allclose(
        scores_from_probabilities(proba, y_train, detector="confidence"),
        confidence_scores(model, X_train, y_train, cv=3, random_state=42),
    )
    assert np.allclose(
        scores_from_probabilities(proba, y_train, detector="margin"),
        margin_scores(model, X_train, y_train, cv=3, random_state=42),
    )


def test_select_top_k_suspicious_validates_fraction():
    with pytest.raises(ValueError):
        select_top_k_suspicious([0.1, 0.2], fraction=1.5)


def test_remove_suspicious_samples_keeps_unflagged_rows():
    dataset = load_dataset("wine")
    suspicious_mask = np.zeros(len(dataset.y), dtype=bool)
    suspicious_mask[:10] = True

    X_clean, y_clean = remove_suspicious_samples(dataset.X, dataset.y, suspicious_mask)

    assert len(X_clean) == len(dataset.X) - 10
    assert len(y_clean) == len(dataset.y) - 10
    assert X_clean.index.equals(dataset.X.index[10:])
    assert y_clean.index.equals(dataset.y.index[10:])


def test_relabel_suspicious_samples_preserves_unflagged_labels():
    dataset = load_dataset("wine")
    X_train, _, _, y_train, _, _ = make_train_val_test_split(
        dataset.X,
        dataset.y,
        random_state=42,
    )
    model = get_baseline_models(random_state=42)["logistic_regression"]
    suspicious_mask = np.zeros(len(y_train), dtype=bool)
    suspicious_mask[:5] = True

    y_relabel = relabel_suspicious_samples(
        model,
        X_train,
        y_train,
        suspicious_mask,
        cv=3,
        random_state=42,
    )

    assert y_relabel.index.equals(y_train.index)
    assert np.all(
        y_relabel.iloc[~suspicious_mask].to_numpy()
        == y_train.iloc[~suspicious_mask].to_numpy()
    )
