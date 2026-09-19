from src.data.load_datasets import load_dataset
from src.data.preprocessing import make_train_val_test_split
import pytest


def test_load_supported_datasets():
    expected_shapes = {
        "breast_cancer": (569, 30, 2),
        "wine": (178, 13, 3),
        "digits": (1797, 64, 10),
    }

    for name, (n_samples, n_features, n_classes) in expected_shapes.items():
        dataset = load_dataset(name)

        assert dataset.X.shape == (n_samples, n_features)
        assert dataset.y.shape == (n_samples,)
        assert dataset.n_classes == n_classes


def test_split_keeps_all_rows_and_stratifies():
    dataset = load_dataset("wine")

    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(
        dataset.X,
        dataset.y,
        random_state=42,
    )

    assert len(X_train) + len(X_val) + len(X_test) == len(dataset.X)
    assert len(y_train) + len(y_val) + len(y_test) == len(dataset.y)
    assert set(y_train.unique()) == set(dataset.y.unique())
    assert set(y_val.unique()) == set(dataset.y.unique())
    assert set(y_test.unique()) == set(dataset.y.unique())
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)


def test_split_is_reproducible_and_rejects_mismatched_lengths():
    dataset = load_dataset("wine")
    first = make_train_val_test_split(dataset.X, dataset.y, random_state=123)
    second = make_train_val_test_split(dataset.X, dataset.y, random_state=123)

    for first_part, second_part in zip(first, second):
        assert first_part.index.equals(second_part.index)

    with pytest.raises(ValueError, match="same number"):
        make_train_val_test_split(dataset.X.iloc[:-1], dataset.y, random_state=123)


def test_dataset_loader_rejects_invalid_names():
    with pytest.raises(ValueError, match="non-empty string"):
        load_dataset(None)
