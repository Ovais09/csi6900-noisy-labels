"""Small I/O helpers used by experiment scripts."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import yaml


def ensure_parent_dir(path) -> Path:
    """Create the parent directory for a file path and return it as a Path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path):
    """Load a YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_dataframe(df: pd.DataFrame, path) -> Path:
    """Save a DataFrame to CSV after creating the output directory."""

    path = ensure_parent_dir(path)
    df.to_csv(path, index=False)
    return path


def save_dataframe_atomic(df: pd.DataFrame, path) -> Path:
    """Atomically replace a CSV so interrupted writes cannot corrupt it."""

    path = ensure_parent_dir(path)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            df.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())

        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return path
