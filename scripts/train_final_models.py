"""Train and save final comparison models for one noisy-label setting."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.experiment_runner import run_single_experiment_details
from src.utils.io import save_dataframe_atomic
from src.utils.provenance import sha256_file

MODEL_CONDITIONS = (
    "clean_upper_bound",
    "noisy_baseline",
    "remove_suspicious",
    "relabel_suspicious",
)


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
    label-noise metadata. It uses the same experiment engine as the main matrix
    while retaining fitted estimators for diagnostics and reproducibility checks.
    """

    details = run_single_experiment_details(
        dataset_name=dataset_name,
        noise_type=noise_type,
        noise_rate=noise_rate,
        model_name=model_name,
        random_state=random_state,
        cv_folds=cv_folds,
        detector_methods=(detector,),
    )
    fitted_models = {
        condition: details["models"][
            (
                condition,
                "none"
                if condition in {"clean_upper_bound", "noisy_baseline"}
                else detector,
            )
        ]
        for condition in MODEL_CONDITIONS
    }
    detector_artifacts = details["detectors"][detector]
    metrics = details["results"].copy()
    metrics.insert(7, "cv_folds", cv_folds)
    metrics.insert(9, "n_validation", len(details["X_validation"]))

    return {
        "dataset": details["dataset"],
        "models": fitted_models,
        "metrics": metrics,
        "X_train": details["X_train"],
        "X_validation": details["X_validation"],
        "X_test": details["X_test"],
        "y_train": details["y_train"],
        "y_train_noisy": details["y_train_noisy"],
        "y_validation": details["y_validation"],
        "y_test": details["y_test"],
        "corrupted_mask": details["corrupted_mask"],
        "suspicious_mask": detector_artifacts["suspicious_mask"],
        "scores": detector_artifacts["scores"],
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
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=output_dir,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
            joblib.dump(model, temporary_path)
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        model_paths.append(
            {
                "condition": condition,
                "model_path": str(path),
                "model_sha256": sha256_file(path),
                "model_bytes": path.stat().st_size,
            }
        )

    model_path_frame = pd.DataFrame(model_paths)
    metadata = metadata.merge(model_path_frame, on="condition", how="left")

    metadata_path = Path(metadata_path) if metadata_path else output_dir / "model_metadata.csv"
    save_dataframe_atomic(metadata, metadata_path)
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
