import hashlib
import json

import pandas as pd
import yaml

from src.utils.config import validate_experiment_config
from src.utils.io import save_json_atomic
from src.utils.provenance import (
    build_run_manifest,
    dataframe_sha256,
    dataset_fingerprint,
    sha256_file,
)


def _config(results_path, summary_path):
    return validate_experiment_config(
        {
            "datasets": ["wine"],
            "noise_types": ["symmetric"],
            "noise_rates": [0.1],
            "models": ["logistic_regression"],
            "detectors": ["confidence"],
            "seeds": [42],
            "cv_folds": 3,
            "output": {
                "results_csv": str(results_path),
                "summary_csv": str(summary_path),
            },
        }
    )


def test_file_and_dataframe_hashes_are_deterministic(tmp_path):
    path = tmp_path / "example.txt"
    path.write_text("reproducible\n", encoding="utf-8")
    expected = hashlib.sha256(b"reproducible\n").hexdigest()

    assert sha256_file(path) == expected

    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert dataframe_sha256(frame) == dataframe_sha256(frame.copy())
    changed = frame.copy()
    changed.loc[0, "a"] = 99
    assert dataframe_sha256(frame) != dataframe_sha256(changed)


def test_dataset_fingerprint_contains_dimensions_and_content_hash():
    fingerprint = dataset_fingerprint("wine")

    assert fingerprint["samples"] == 178
    assert fingerprint["features"] == 13
    assert fingerprint["classes"] == 3
    assert len(fingerprint["content_sha256"]) == 64


def test_build_and_save_run_manifest(tmp_path):
    results_path = tmp_path / "results.csv"
    pd.DataFrame({"accuracy": [0.9], "condition": ["clean"]}).to_csv(
        results_path,
        index=False,
    )
    config_path = tmp_path / "config.yaml"
    config = _config(results_path, tmp_path / "summary.csv")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    manifest = build_run_manifest(
        config=config,
        config_path=config_path,
        results_path=results_path,
        project_root=tmp_path,
    )
    output_path = save_json_atomic(manifest, tmp_path / "manifest.json")
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["schema_version"] == 1
    assert saved["results"]["rows"] == 1
    assert saved["results"]["sha256"] == sha256_file(results_path)
    assert saved["configuration"]["sha256"] == sha256_file(config_path)
    assert saved["datasets"][0]["name"] == "wine"
    assert saved["runtime"]["dependencies"]["scikit-learn"]
    assert saved["git"] == {"commit": None, "dirty": None}
