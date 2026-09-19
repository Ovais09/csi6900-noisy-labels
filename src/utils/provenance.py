"""Reproducibility fingerprints and run-manifest helpers."""

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from src.data.load_datasets import load_dataset


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "pyyaml",
    "joblib",
)


def sha256_file(path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(df: pd.DataFrame) -> str:
    """Hash a dataframe using deterministic CSV serialization."""

    payload = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_fingerprint(dataset_name: str) -> dict:
    """Return dimensions and a content hash for one supported dataset."""

    dataset = load_dataset(dataset_name)
    frame = dataset.X.copy()
    frame["target"] = dataset.y.to_numpy()
    return {
        "name": dataset.name,
        "samples": len(dataset.y),
        "features": dataset.X.shape[1],
        "classes": dataset.n_classes,
        "task_type": dataset.task_type,
        "content_sha256": dataframe_sha256(frame),
    }


def dependency_versions() -> dict:
    """Return versions of the packages that affect experiment behavior."""

    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def git_state(project_root) -> dict:
    """Return Git revision metadata, tolerating Docker builds without `.git`."""

    project_root = Path(project_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}

    return {"commit": commit, "dirty": bool(status.strip())}


def build_run_manifest(config, config_path, results_path, project_root) -> dict:
    """Build a complete provenance record for a finished result CSV."""

    config_path = Path(config_path)
    results_path = Path(results_path)
    results = pd.read_csv(results_path)

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(project_root),
        "runtime": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "configuration": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "values": config,
        },
        "datasets": [
            dataset_fingerprint(dataset_name) for dataset_name in config["datasets"]
        ],
        "results": {
            "path": str(results_path),
            "sha256": sha256_file(results_path),
            "rows": len(results),
            "columns": list(results.columns),
        },
    }
