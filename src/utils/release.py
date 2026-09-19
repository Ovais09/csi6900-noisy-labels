"""Immutable checksum manifests for final experiment artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.provenance import git_state, sha256_file


def _relative_file(path, project_root: Path) -> tuple[Path, str]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root / resolved
    resolved = resolved.resolve()

    try:
        relative = resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"Artifact is outside the project root: {path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact is missing or is not a file: {relative}")
    return resolved, relative.as_posix()


def build_release_manifest(paths, project_root, metadata=None) -> dict:
    """Create a deterministic file inventory with content hashes."""

    project_root = Path(project_root).resolve()
    records = []
    seen = set()
    for path in paths:
        resolved, relative = _relative_file(path, project_root)
        if relative in seen:
            continue
        seen.add(relative)
        records.append(
            {
                "path": relative,
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )

    records.sort(key=lambda record: record["path"])
    if not records:
        raise ValueError("A release manifest must contain at least one artifact")

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(project_root),
        "artifact_count": len(records),
        "metadata": metadata or {},
        "artifacts": records,
    }


def load_release_manifest(path) -> dict:
    """Load a release manifest from JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_release_manifest(manifest, project_root, expected_paths=None) -> list[dict]:
    """Verify manifest structure, revision, file sizes, and SHA-256 hashes."""

    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or malformed release manifest")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("Release manifest has no artifact records")
    if manifest.get("artifact_count") != len(records):
        raise ValueError("Release manifest artifact_count is inconsistent")

    project_root = Path(project_root).resolve()
    paths = [record.get("path") for record in records]
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("Release manifest contains an invalid artifact path")
    if len(paths) != len(set(paths)):
        raise ValueError("Release manifest contains duplicate artifact paths")

    if expected_paths is not None:
        expected = {
            _relative_file(path, project_root)[1]
            for path in expected_paths
        }
        actual = set(paths)
        if actual != expected:
            raise ValueError(
                "Release manifest artifact set differs from the canonical set: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )

    frozen_commit = manifest.get("git", {}).get("commit")
    current_commit = git_state(project_root).get("commit")
    if frozen_commit != current_commit:
        raise ValueError(
            "Release manifest Git revision differs from the current revision: "
            f"frozen={frozen_commit}, current={current_commit}"
        )

    for record in records:
        resolved, relative = _relative_file(record["path"], project_root)
        expected_bytes = record.get("bytes")
        expected_hash = record.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f"Invalid byte count for artifact: {relative}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Invalid SHA-256 value for artifact: {relative}")
        if resolved.stat().st_size != expected_bytes:
            raise ValueError(f"Artifact size changed after freezing: {relative}")
        if sha256_file(resolved) != expected_hash:
            raise ValueError(f"Artifact hash changed after freezing: {relative}")

    return records
