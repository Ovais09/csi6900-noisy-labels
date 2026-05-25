"""Core experiment orchestration."""

import pandas as pd
from sklearn.base import clone

from src.cleaning.cleaning_methods import (
    relabel_suspicious_samples,
    remove_suspicious_samples,
    select_top_k_suspicious,
)
from src.cleaning.detectors import suspicious_scores
from src.data.load_datasets import load_dataset
from src.data.preprocessing import make_train_val_test_split
from src.evaluation.metrics import classification_metrics, detection_metrics
from src.models.baselines import get_baseline_models
from src.noise.inject_noise import (
    inject_class_dependent_noise,
    inject_symmetric_noise,
)


def _inject_noise(y_train, noise_type: str, noise_rate: float, random_state: int):
    if noise_type == "symmetric":
        return inject_symmetric_noise(
            y_train,
            noise_rate=noise_rate,
            random_state=random_state,
        )

    if noise_type == "class_dependent":
        return inject_class_dependent_noise(
            y_train,
            noise_rate=noise_rate,
            random_state=random_state,
        )

    raise ValueError(f"Unknown noise type: {noise_type}")


def _base_row(
    dataset_name,
    noise_type,
    noise_rate,
    model_name,
    condition,
    detector,
    random_state,
    n_train,
    n_test,
    n_corrupted,
    n_suspicious,
):
    return {
        "dataset": dataset_name,
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "model": model_name,
        "condition": condition,
        "detector": detector,
        "random_state": random_state,
        "n_train": n_train,
        "n_test": n_test,
        "n_corrupted": n_corrupted,
        "n_suspicious": n_suspicious,
    }


def run_single_experiment(
    dataset_name,
    noise_type,
    noise_rate,
    model_name,
    random_state: int = 42,
    cv_folds: int = 5,
    detector_methods=None,
):
    """Run clean, noisy, remove-cleaned, and relabeled conditions."""

    if detector_methods is None:
        detector_methods = ("combined",)
    elif isinstance(detector_methods, str):
        detector_methods = (detector_methods,)
    else:
        detector_methods = tuple(detector_methods)

    dataset = load_dataset(dataset_name)
    X_train, _, X_test, y_train, _, y_test = make_train_val_test_split(
        dataset.X,
        dataset.y,
        random_state=random_state,
    )
    y_train_noisy, corrupted_mask = _inject_noise(
        y_train,
        noise_type=noise_type,
        noise_rate=noise_rate,
        random_state=random_state,
    )

    models = get_baseline_models(random_state=random_state)
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    base_model = models[model_name]
    rows = []
    common_counts = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_corrupted": int(corrupted_mask.sum()),
    }

    clean_model = clone(base_model)
    clean_model.fit(X_train, y_train)
    clean_metrics = classification_metrics(clean_model, X_test, y_test)
    rows.append(
        {
            **_base_row(
                dataset_name,
                noise_type,
                noise_rate,
                model_name,
                "clean_upper_bound",
                "none",
                random_state,
                **common_counts,
                n_suspicious=None,
            ),
            **clean_metrics,
            "det_precision": None,
            "det_recall": None,
            "det_f1": None,
        }
    )

    noisy_model = clone(base_model)
    noisy_model.fit(X_train, y_train_noisy)
    noisy_metrics = classification_metrics(noisy_model, X_test, y_test)
    rows.append(
        {
            **_base_row(
                dataset_name,
                noise_type,
                noise_rate,
                model_name,
                "noisy_baseline",
                "none",
                random_state,
                **common_counts,
                n_suspicious=None,
            ),
            **noisy_metrics,
            "det_precision": None,
            "det_recall": None,
            "det_f1": None,
        }
    )

    for detector in detector_methods:
        scores = suspicious_scores(
            base_model,
            X_train,
            y_train_noisy,
            detector=detector,
            cv=cv_folds,
            random_state=random_state,
        )
        suspicious_mask = select_top_k_suspicious(scores, fraction=noise_rate)
        det_metrics = detection_metrics(corrupted_mask, suspicious_mask)
        n_suspicious = int(suspicious_mask.sum())

        X_train_removed, y_train_removed = remove_suspicious_samples(
            X_train,
            y_train_noisy,
            suspicious_mask,
        )
        removed_model = clone(base_model)
        removed_model.fit(X_train_removed, y_train_removed)
        removed_metrics = classification_metrics(removed_model, X_test, y_test)
        rows.append(
            {
                **_base_row(
                    dataset_name,
                    noise_type,
                    noise_rate,
                    model_name,
                    "remove_suspicious",
                    detector,
                    random_state,
                    **common_counts,
                    n_suspicious=n_suspicious,
                ),
                **removed_metrics,
                **det_metrics,
            }
        )

        y_train_relabel = relabel_suspicious_samples(
            base_model,
            X_train,
            y_train_noisy,
            suspicious_mask,
            cv=cv_folds,
            random_state=random_state,
        )
        relabel_model = clone(base_model)
        relabel_model.fit(X_train, y_train_relabel)
        relabel_metrics = classification_metrics(relabel_model, X_test, y_test)
        rows.append(
            {
                **_base_row(
                    dataset_name,
                    noise_type,
                    noise_rate,
                    model_name,
                    "relabel_suspicious",
                    detector,
                    random_state,
                    **common_counts,
                    n_suspicious=n_suspicious,
                ),
                **relabel_metrics,
                **det_metrics,
            }
        )

    return pd.DataFrame(rows)
