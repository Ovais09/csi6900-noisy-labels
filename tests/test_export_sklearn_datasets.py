import pandas as pd

from scripts.export_sklearn_datasets import export_datasets


def test_export_datasets_writes_csv_and_manifest(tmp_path):
    manifest = export_datasets(["wine"], output_dir=tmp_path)
    dataset_path = tmp_path / "wine.csv"
    manifest_path = tmp_path / "manifest.csv"

    assert dataset_path.exists()
    assert manifest_path.exists()

    df = pd.read_csv(dataset_path)
    saved_manifest = pd.read_csv(manifest_path)

    assert df.shape == (178, 14)
    assert "target" in df.columns
    assert manifest.loc[0, "dataset"] == "wine"
    assert saved_manifest.loc[0, "dataset"] == "wine"
    assert saved_manifest.loc[0, "samples"] == 178
    assert len(saved_manifest.loc[0, "sha256"]) == 64
