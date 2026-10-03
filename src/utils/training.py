"""Bind result checkpoints to the code, data, settings, and runtime that trained them."""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils.io import load_dataframe, save_json_atomic
from src.utils.provenance import (
    dataset_fingerprint,
    dependency_versions,
    git_state,
    sha256_file,
)


def training_manifest_path(results_path):
    return Path(str(results_path) + ".training.json")


def training_source_hashes(project_root):
    """Fingerprint the implementation that determines fitted models and metrics."""
    root = Path(project_root)
    files = []
    for package in ("data", "noise", "models", "cleaning"):
        files.extend((root / "src" / package).glob("*.py"))
    files.extend(root / path for path in (
        "src/evaluation/experiment_runner.py",
        "src/evaluation/metrics.py",
        "src/utils/seeds.py",
        "scripts/run_cleaning_experiments.py",
    ))
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(files)
    }


def training_inputs(config, project_root):
    return {
        "configuration": {key: value for key, value in config.items() if key != "output"},
        "sources": training_source_hashes(project_root),
        "datasets": [dataset_fingerprint(name) for name in config["datasets"]],
    }


def training_runtime():
    return {
        "python": sys.version.split()[0],
        "system": platform.system(),
        "machine": platform.machine(),
        "dependencies": dependency_versions(),
    }


def verify_training_record(config, results_path, project_root, *, for_resume=False):
    """Reject untraceable, modified, or incompatible result reuse."""
    path = training_manifest_path(results_path)
    if not path.is_file():
        raise ValueError(f"Missing training provenance: {path}. Run training without --resume.")
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("Unsupported training provenance schema")
    if record.get("inputs") != training_inputs(config, project_root):
        raise ValueError("Training code, configuration, or dataset changed; start a fresh run.")
    checkpoint = record.get("results") or {}
    if checkpoint.get("sha256") != sha256_file(results_path):
        raise ValueError("Result checkpoint hash differs from training provenance; start a fresh run.")
    if checkpoint.get("rows") != len(load_dataframe(results_path)):
        raise ValueError("Result checkpoint row count differs from training provenance")
    if for_resume and record.get("runtime") != training_runtime():
        raise ValueError("Training runtime changed; resume in the original environment or start fresh.")
    if not for_resume and record.get("complete") is not True:
        raise ValueError("Training has not completed; resume training before publishing results.")
    return record


def start_training_record(config, results_path, project_root, *, resume=False):
    if resume and Path(results_path).exists():
        return verify_training_record(config, results_path, project_root, for_resume=True)
    record = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(project_root),
        "inputs": training_inputs(config, project_root),
        "runtime": training_runtime(),
        "complete": False,
        "results": None,
    }
    save_json_atomic(record, training_manifest_path(results_path))
    return record


def checkpoint_training(record, results_path, *, complete=False):
    record = dict(record)
    record["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    record["complete"] = complete
    record["results"] = {
        "sha256": sha256_file(results_path),
        "rows": len(load_dataframe(results_path)),
    }
    save_json_atomic(record, training_manifest_path(results_path))
    return record
