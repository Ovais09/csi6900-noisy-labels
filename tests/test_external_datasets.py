from zipfile import ZipFile

import pandas as pd

from src.data.external_datasets import parse_dry_bean_arff, prepare_dry_bean_dataset


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
