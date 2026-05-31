"""Train and save final comparison models for one noisy-label setting."""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cleaning.cleaning_methods import (
    remove_suspicious_samples,
    select_top_k_suspicious,
)
from src.cleaning.detectors import (
    cross_validated_probabilities,
    predicted_labels_from_probabilities,
    scores_from_probabilities,
)
from src.data.load_datasets import load_dataset
from src.data.preprocessing import make_train_val_test_split
from src.evaluation.metrics import classification_metrics, detection_metrics
from src.models.baselines import get_baseline_models
from src.noise.inject_noise import (
    inject_class_dependent_noise,
    inject_symmetric_noise,
)
from src.utils.io import save_dataframe


MODEL_CONDITIONS = (
    "clean_upper_bound",
    "noisy_baseline",
    "remove_suspicious",
    "relabel_suspicious",
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


def train_final_model_set(
    dataset_name: str,
    model_name: str,
    noise_type: str,
    noise_rate: float,
    detector: str,
    random_state: int = 42,
    cv_folds: int = 5,
):
    """Train clean, noisy, remove-cleaned, and relabeled models.

    Returns a dictionary containing fitted models, metrics, split data, and
    label-noise metadata. This is intentionally separate from the main
    experiment runner because the main runner saves scalar metrics, while this
    function keeps fitted estimators for diagnostics and reproducibility checks.
    """

    dataset = load_dataset(dataset_name)
    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(
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
    fitted_models = {}
    rows = []

    clean_model = clone(base_model)
    clean_model.fit(X_train, y_train)
    fitted_models["clean_upper_bound"] = clean_model

    noisy_model = clone(base_model)
    noisy_model.fit(X_train, y_train_noisy)
    fitted_models["noisy_baseline"] = noisy_model

    cv_proba = cross_validated_probabilities(
        base_model,
        X_train,
        y_train_noisy,
        cv=cv_folds,
        random_state=random_state,
    )
    cv_pred_labels = predicted_labels_from_probabilities(cv_proba, y_train_noisy)
    scores = scores_from_probabilities(cv_proba, y_train_noisy, detector=detector)
    suspicious_mask = select_top_k_suspicious(scores, fraction=noise_rate)
    det_metrics = detection_metrics(corrupted_mask, suspicious_mask)

    X_train_removed, y_train_removed = remove_suspicious_samples(
        X_train,
        y_train_noisy,
        suspicious_mask,
    )
    removed_model = clone(base_model)
    removed_model.fit(X_train_removed, y_train_removed)
    fitted_models["remove_suspicious"] = removed_model

    y_train_relabel = y_train_noisy.copy()
    y_train_relabel.iloc[suspicious_mask] = cv_pred_labels[suspicious_mask]
    relabel_model = clone(base_model)
    relabel_model.fit(X_train, y_train_relabel)
    fitted_models["relabel_suspicious"] = relabel_model

    for condition, model in fitted_models.items():
        metrics = classification_metrics(model, X_test, y_test)
        row = {
            "dataset": dataset_name,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "model": model_name,
            "condition": condition,
            "detector": "none" if condition in {"clean_upper_bound", "noisy_baseline"} else detector,
            "random_state": random_state,
            "cv_folds": cv_folds,
            "n_train": len(X_train),
            "n_validation": len(X_val),
            "n_test": len(X_test),
            "n_corrupted": int(corrupted_mask.sum()),
            "n_suspicious": None
            if condition in {"clean_upper_bound", "noisy_baseline"}
            else int(suspicious_mask.sum()),
            **metrics,
        }
        if condition in {"remove_suspicious", "relabel_suspicious"}:
            row.update(det_metrics)
        else:
            row.update({"det_precision": None, "det_recall": None, "det_f1": None})
        rows.append(row)

    return {
        "dataset": dataset,
        "models": fitted_models,
        "metrics": pd.DataFrame(rows),
        "X_train": X_train,
        "X_validation": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_train_noisy": y_train_noisy,
        "y_validation": y_val,
        "y_test": y_test,
        "corrupted_mask": corrupted_mask,
        "suspicious_mask": suspicious_mask,
        "scores": scores,
    }


def model_artifact_dir(
    output_dir,
    dataset_name: str,
    model_name: str,
    noise_type: str,
    noise_rate: float,
    detector: str,
    random_state: int,
) -> Path:
    """Return a stable directory for one saved model set."""

    rate_label = f"{noise_rate:g}".replace(".", "p")
    return (
        Path(output_dir)
        / dataset_name
        / f"{noise_type}_rate_{rate_label}"
        / model_name
        / detector
        / f"seed_{random_state}"
    )


def save_final_model_set(artifacts, output_dir, metadata_path=None):
    """Persist fitted models and a metadata CSV."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = artifacts["metrics"].copy()
    model_paths = []
    for condition, model in artifacts["models"].items():
        path = output_dir / f"{condition}.joblib"
        joblib.dump(model, path)
        model_paths.append({"condition": condition, "model_path": str(path)})

    model_path_frame = pd.DataFrame(model_paths)
    metadata = metadata.merge(model_path_frame, on="condition", how="left")

    metadata_path = Path(metadata_path) if metadata_path else output_dir / "model_metadata.csv"
    save_dataframe(metadata, metadata_path)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dry_bean")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--noise-type", default="symmetric")
    parser.add_argument("--noise-rate", type=float, default=0.2)
    parser.add_argument("--detector", default="confidence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="results/models")
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts = train_final_model_set(
        dataset_name=args.dataset,
        model_name=args.model,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        detector=args.detector,
        random_state=args.seed,
        cv_folds=args.cv_folds,
    )
    output_dir = model_artifact_dir(
        args.output_dir,
        dataset_name=args.dataset,
        model_name=args.model,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        detector=args.detector,
        random_state=args.seed,
    )
    metadata = save_final_model_set(artifacts, output_dir)

    print(f"Saved models to {output_dir}")
    print(metadata.to_string(index=False))


if __name__ == "__main__":
    main()
