"""Dataset loaders for the noisy-label benchmark."""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

SUPPORTED_DATASETS = ("breast_cancer", "wine", "digits", "dry_bean")


@dataclass(frozen=True)
class DatasetBundle:
    """Container for a supervised tabular classification dataset."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    task_type: str
    n_classes: int


def _bundle_from_sklearn(name: str, loader, task_type: str) -> DatasetBundle:
    data = loader(as_frame=True)
    X = data.data
    y = data.target.astype(int)

    return DatasetBundle(
        name=name,
        X=X,
        y=y,
        task_type=task_type,
        n_classes=int(y.nunique()),
    )


def load_dataset(name: str) -> DatasetBundle:
    """Load a supported dataset by name.

    Parameters
    ----------
    name:
        One of ``breast_cancer``, ``wine``, ``digits``, or ``dry_bean``.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Dataset name must be a non-empty string")
    normalized_name = name.lower()

    if normalized_name == "breast_cancer":
        return _bundle_from_sklearn(
            name="breast_cancer",
            loader=load_breast_cancer,
            task_type="binary",
        )

    if normalized_name == "wine":
        return _bundle_from_sklearn(
            name="wine",
            loader=load_wine,
            task_type="multiclass",
        )

    if normalized_name == "digits":
        return _bundle_from_sklearn(
            name="digits",
            loader=load_digits,
            task_type="multiclass",
        )

    if normalized_name == "dry_bean":
        from src.data.external_datasets import load_dry_bean

        return load_dry_bean()

    raise ValueError(f"Unknown dataset: {name}")
