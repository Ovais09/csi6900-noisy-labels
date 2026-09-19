from zipfile import ZipFile

import pandas as pd

import pytest

from src.data.external_datasets import (
    download_dry_bean,
    load_dry_bean,
    parse_dry_bean_arff,
    prepare_dry_bean_dataset,
)


TINY_DRY_BEAN_ARFF = """@RELATION DryBeanDataset

@ATTRIBUTE Area NUMERIC
@ATTRIBUTE Perimeter NUMERIC
@ATTRIBUTE Class {SEKER,BARBUNYA}

@DATA
100,40.5,SEKER
120,44.1,BARBUNYA
130,45.2,SEKER
"""


def test_parse_dry_bean_arff_encodes_target(tmp_path):
    arff_path = tmp_path / "Dry_Bean_Dataset.arff"
    arff_path.write_text(TINY_DRY_BEAN_ARFF, encoding="utf-8")

    df, mapping = parse_dry_bean_arff(arff_path)

    assert df.shape == (3, 3)
    assert list(df.columns) == ["Area", "Perimeter", "target"]
    assert set(df["target"]) == {0, 1}
    assert set(mapping["class_label"]) == {"BARBUNYA", "SEKER"}


def test_prepare_dry_bean_dataset_from_local_zip(tmp_path):
    zip_path = tmp_path / "DryBeanDataset.zip"
    arff_member = "DryBeanDataset/Dry_Bean_Dataset.arff"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr(arff_member, TINY_DRY_BEAN_ARFF)

    processed_path = tmp_path / "processed" / "dry_bean.csv"
    mapping_path = tmp_path / "processed" / "dry_bean_label_mapping.csv"

    output_path = prepare_dry_bean_dataset(
        zip_path=zip_path,
        raw_dir=tmp_path / "raw",
        processed_path=processed_path,
        label_mapping_path=mapping_path,
    )

    assert output_path == processed_path
    assert processed_path.exists()
    assert mapping_path.exists()

    df = pd.read_csv(processed_path)
    mapping = pd.read_csv(mapping_path)

    assert df.shape == (3, 3)
    assert set(df["target"]) == {0, 1}
    assert set(mapping["class_label"]) == {"BARBUNYA", "SEKER"}


def test_parse_dry_bean_rejects_missing_class_attribute(tmp_path):
    arff_path = tmp_path / "invalid.arff"
    arff_path.write_text(
        """@RELATION invalid
@ATTRIBUTE Area NUMERIC
@DATA
100
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Class"):
        parse_dry_bean_arff(arff_path)


def test_download_dry_bean_publishes_only_valid_zip(tmp_path, monkeypatch):
    destination = tmp_path / "DryBeanDataset.zip"

    def fake_download(_url, temporary_path):
        with ZipFile(temporary_path, "w") as archive:
            archive.writestr(
                "DryBeanDataset/Dry_Bean_Dataset.arff",
                TINY_DRY_BEAN_ARFF,
            )

    monkeypatch.setattr("src.data.external_datasets.urlretrieve", fake_download)

    result = download_dry_bean(destination, url="https://example.invalid/data.zip")

    assert result == destination
    assert destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_download_dry_bean_rejects_archive_without_dataset(tmp_path, monkeypatch):
    destination = tmp_path / "DryBeanDataset.zip"

    def fake_download(_url, temporary_path):
        with ZipFile(temporary_path, "w") as archive:
            archive.writestr("unrelated.txt", "not the dataset")

    monkeypatch.setattr("src.data.external_datasets.urlretrieve", fake_download)

    with pytest.raises(ValueError, match="expected ARFF"):
        download_dry_bean(destination, url="https://example.invalid/data.zip")

    assert not destination.exists()


def test_load_dry_bean_rejects_invalid_cached_targets(tmp_path, monkeypatch):
    processed_path = tmp_path / "dry_bean.csv"
    pd.DataFrame(
        {
            "Area": [100.0, 120.0, 130.0],
            "target": [0, 1.5, 1],
        }
    ).to_csv(processed_path, index=False)
    monkeypatch.setattr(
        "src.data.external_datasets.DRY_BEAN_PROCESSED_PATH",
        processed_path,
    )

    with pytest.raises(ValueError, match="integer class IDs"):
        load_dry_bean(auto_prepare=False)
