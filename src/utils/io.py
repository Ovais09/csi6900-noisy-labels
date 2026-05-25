"""Small I/O helpers used by experiment scripts."""

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
