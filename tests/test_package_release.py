import io
import json
import subprocess
import tarfile

import pytest

from scripts.package_release import BUNDLE_INDEX, create_bundle, verify_bundle
from src.utils.io import save_json_atomic
from src.utils.provenance import sha256_file
from src.utils.release import build_release_manifest


@pytest.fixture
def frozen_checkout(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("results/\n", encoding="utf-8")
    (root / "main.py").write_text("print('ready')\n", encoding="utf-8")
    for command in (["init"], ["add", "."],
                    ["-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                     "commit", "-m", "Fixture"]):
        subprocess.run(["git", *command], cwd=root, check=True, capture_output=True)
    artifact = root / "results" / "table.csv"
    artifact.parent.mkdir()
    artifact.write_text("accuracy\n0.9\n", encoding="utf-8")
    release_path = root / "results/logs/release_manifest.json"
    save_json_atomic(build_release_manifest([artifact], root), release_path)
    save_json_atomic({"status": "PASS", "release_manifest_sha256": sha256_file(release_path)},
                     root / "results/logs/final_audit.json")
    return root


def test_release_bundle_includes_source_and_ignored_results_and_detects_tampering(tmp_path, frozen_checkout):
    root = frozen_checkout
    bundle = tmp_path / "release.tar.gz"
    index = create_bundle(bundle, root)
    assert verify_bundle(bundle) == index
    names = {record["path"] for record in index["files"]}
    assert {"main.py", "results/table.csv", "results/logs/final_audit.json"} <= names
    assert not any(name.startswith(".git/") for name in names)

    corrupt = tmp_path / "corrupt.tar.gz"
    with tarfile.open(bundle, "r:gz") as source, tarfile.open(corrupt, "w:gz") as target:
        for member in source.getmembers():
            payload = source.extractfile(member).read()
            if member.name == "results/table.csv":
                payload = payload.replace(b"0.9", b"0.1")
            target.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bundle(corrupt)
    (root / "main.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean Git checkout"):
        create_bundle(bundle, root)


@pytest.mark.parametrize("output_path", [
    "main.py", "results/table.csv", "results/logs/release_manifest.json",
    "results/logs/final_audit.json",
])
def test_bundle_output_cannot_replace_a_source_or_release_input(frozen_checkout, output_path):
    output = frozen_checkout / output_path
    original = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite a release input"):
        create_bundle(output, frozen_checkout)
    assert output.read_bytes() == original


def test_bundle_output_alias_cannot_replace_a_release_input(frozen_checkout):
    output = frozen_checkout / "results" / "alias.tar.gz"
    artifact = frozen_checkout / "results" / "table.csv"
    original = artifact.read_bytes()
    output.symlink_to(artifact)
    with pytest.raises(ValueError, match="overwrite a release input"):
        create_bundle(output, frozen_checkout)
    assert artifact.read_bytes() == original


@pytest.mark.parametrize("name", ["../outside", "/absolute"])
def test_bundle_rejects_unsafe_members_without_extracting(tmp_path, name):
    bundle = tmp_path / "unsafe.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        archive.addfile(member, io.BytesIO())
    with pytest.raises(ValueError, match="Unsafe bundle member"):
        verify_bundle(bundle)


def test_bundle_rejects_unlisted_members(tmp_path):
    bundle = tmp_path / "extra.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        data = json.dumps({"schema_version": 1, "files": []}).encode()
        member = tarfile.TarInfo(BUNDLE_INDEX)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
        archive.addfile(tarfile.TarInfo("unexpected"), io.BytesIO())
    with pytest.raises(ValueError, match="differ from its inventory"):
        verify_bundle(bundle)
