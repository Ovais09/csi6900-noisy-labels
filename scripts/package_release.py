"""Package a frozen release with its exact source and verify it without extraction."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.provenance import git_state, sha256_file
from src.utils.release import load_release_manifest, verify_release_manifest

BUNDLE_INDEX = "BUNDLE_MANIFEST.json"


def verify_bundle(path):
    """Check every archived byte and reject duplicate, unsafe, or extra members."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Bundle contains duplicate member names")
        for member in members:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe bundle member: {member.name}")
        if BUNDLE_INDEX not in names:
            raise ValueError("Bundle is missing its checksum inventory")
        with archive.extractfile(BUNDLE_INDEX) as handle:
            index = json.load(handle)
        if index.get("schema_version") != 1 or not isinstance(index.get("files"), list):
            raise ValueError("Unsupported bundle inventory")
        records = index["files"]
        expected = [record["path"] for record in records]
        if len(expected) != len(set(expected)) or set(names) != {*expected, BUNDLE_INDEX}:
            raise ValueError("Bundle members differ from its inventory")
        for record in records:
            member = archive.getmember(record["path"])
            digest = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if member.size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Bundle checksum mismatch: {member.name}")
    return index


def create_bundle(output, project_root=PROJECT_ROOT,
                  release_path="results/logs/release_manifest.json",
                  audit_path="results/logs/final_audit.json"):
    root = Path(project_root).resolve()
    state = git_state(root)
    if state["dirty"] is not False or not state["commit"]:
        raise ValueError("Package from a clean Git checkout after freezing the release")
    manifest = load_release_manifest(root / release_path)
    artifacts = verify_release_manifest(manifest, root)
    with (root / audit_path).open(encoding="utf-8") as handle:
        audit = json.load(handle)
    if (audit.get("status") != "PASS"
            or audit.get("release_manifest_sha256") != sha256_file(root / release_path)):
        raise ValueError("A passing audit of this exact release manifest is required")
    source = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True,
                            capture_output=True).stdout.decode().split("\0")
    paths = sorted(set(filter(None, source)) | {r["path"] for r in artifacts}
                   | {str(release_path), str(audit_path)})
    records = []
    for name in paths:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Cannot package a linked or external file: {name}")
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    index = {"schema_version": 1, "source_commit": state["commit"], "files": records}
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tar.gz", delete=False) as handle:
            temporary = Path(handle.name)
        with tarfile.open(temporary, "w:gz") as archive:
            for name in paths:
                archive.add(root / name, arcname=name, recursive=False)
            payload = json.dumps(index, indent=2).encode("utf-8")
            member = tarfile.TarInfo(BUNDLE_INDEX)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        verify_bundle(temporary)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/logs/csi6900-final-release.tar.gz")
    parser.add_argument("--verify", metavar="ARCHIVE", help="Verify an existing bundle only")
    args = parser.parse_args()
    index = verify_bundle(args.verify) if args.verify else create_bundle(args.output)
    path = args.verify or args.output
    print(f"Verified {len(index['files'])} files at commit {index['source_commit']}")
    print(f"{sha256_file(path)}  {path}")


if __name__ == "__main__":
    main()
