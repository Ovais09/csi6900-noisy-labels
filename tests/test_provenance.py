import hashlib
import json

import pandas as pd
import pytest
import yaml

from scripts.audit_release import _check_run_manifest
from src.utils.config import validate_experiment_config
from src.utils.io import save_json_atomic
from src.utils.provenance import (
    build_run_manifest,
    dataframe_sha256,
    dataset_fingerprint,
    git_state,
    sha256_file,
)
from src.utils.training import (
    checkpoint_training,
    start_training_record,
    verify_training_record,
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


def test_build_and_save_run_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("CSI6900_GIT_COMMIT", raising=False)
    monkeypatch.setattr("src.utils.training.training_source_hashes", lambda _: {"model.py": "abc"})
    results_path = tmp_path / "results.csv"
    pd.DataFrame({"accuracy": [0.9], "condition": ["clean"]}).to_csv(
        results_path,
        index=False,
    )
    config_path = tmp_path / "config.yaml"
    config = _config(results_path, tmp_path / "summary.csv")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    training = start_training_record(config, results_path, tmp_path)
    checkpoint_training(training, results_path, complete=True)

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
    assert saved["training"]["results"]["sha256"] == sha256_file(results_path)


def test_git_state_uses_container_build_revision_without_git(tmp_path, monkeypatch):
    revision = "abc123def456"
    monkeypatch.setenv("CSI6900_GIT_COMMIT", revision)

    assert git_state(tmp_path) == {"commit": revision, "dirty": None}


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.training.training_source_hashes", lambda _: {"model.py": "abc"})
    results_path = tmp_path / "results.csv"
    pd.DataFrame({"accuracy": [0.9]}).to_csv(results_path, index=False)
    config = _config(results_path, tmp_path / "summary.csv")
    record = start_training_record(config, results_path, tmp_path)
    checkpoint_training(record, results_path, complete=True)
    return config, results_path, tmp_path


def test_resume_verifies_the_original_training_inputs(checkpoint):
    config, results, root = checkpoint
    original = verify_training_record(config, results, root)
    resumed = start_training_record(config, results, root, resume=True)
    assert resumed == original


@pytest.mark.parametrize("change", ["cv", "source", "data", "runtime", "csv"])
def test_resume_rejects_incompatible_or_modified_results(checkpoint, monkeypatch, change):
    config, results, root = checkpoint
    if change == "cv":
        config["cv_folds"] = 5
    elif change == "source":
        monkeypatch.setattr("src.utils.training.training_source_hashes", lambda _: {"model.py": "changed"})
    elif change == "data":
        monkeypatch.setattr("src.utils.training.dataset_fingerprint", lambda _: {"changed": True})
    elif change == "runtime":
        monkeypatch.setattr("src.utils.training.training_runtime", lambda: {"python": "different"})
    else:
        results.write_text(results.read_text().replace("0.9", "0.8"))
    with pytest.raises(ValueError, match="changed|differs"):
        start_training_record(config, results, root, resume=True)


def test_untracked_or_incomplete_results_cannot_be_published(checkpoint):
    config, results, root = checkpoint
    record = verify_training_record(config, results, root)
    checkpoint_training(record, results, complete=False)
    with pytest.raises(ValueError, match="not completed"):
        verify_training_record(config, results, root)
    from src.utils.training import training_manifest_path
    training_manifest_path(results).unlink()
    with pytest.raises(ValueError, match="Missing training provenance"):
        start_training_record(config, results, root, resume=True)


def test_audit_rejects_matching_unknown_source_revisions(checkpoint, monkeypatch):
    config, results, root = checkpoint
    monkeypatch.delenv("CSI6900_GIT_COMMIT", raising=False)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = build_run_manifest(config, config_path, results, root)
    manifest_path = save_json_atomic(manifest, root / "run_manifest.json")
    assert manifest["git"]["commit"] is None
    with pytest.raises(ValueError, match="known source revision"):
        _check_run_manifest(manifest_path, config_path, results, config, root)


def test_audit_rejects_missing_original_training_revision(checkpoint, monkeypatch):
    config, results, root = checkpoint
    monkeypatch.setenv("CSI6900_GIT_COMMIT", "a" * 40)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = build_run_manifest(config, config_path, results, root)
    manifest["training"]["git"]["commit"] = None
    from src.utils.training import training_manifest_path
    save_json_atomic(manifest["training"], training_manifest_path(results))
    manifest_path = save_json_atomic(manifest, root / "run_manifest.json")
    with pytest.raises(ValueError, match="original training revision"):
        _check_run_manifest(manifest_path, config_path, results, config, root)
