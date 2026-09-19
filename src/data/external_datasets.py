"""External public tabular datasets used beyond the sklearn smoke datasets."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

from src.data.load_datasets import DatasetBundle
from src.utils.io import load_dataframe, save_dataframe_atomic

DRY_BEAN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00602/DryBeanDataset.zip"
)
DRY_BEAN_RAW_DIR = Path("datasets/raw/dry_bean")
DRY_BEAN_ZIP_PATH = DRY_BEAN_RAW_DIR / "DryBeanDataset.zip"
DRY_BEAN_ARFF_MEMBER = "DryBeanDataset/Dry_Bean_Dataset.arff"
DRY_BEAN_PROCESSED_PATH = Path("datasets/processed/dry_bean.csv")
DRY_BEAN_LABEL_MAPPING_PATH = Path("datasets/processed/dry_bean_label_mapping.csv")


def _validate_dry_bean_archive(zip_path) -> None:
    try:
        with ZipFile(zip_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(
                    f"Dry Bean archive has a corrupt member: {bad_member}"
                )
            if DRY_BEAN_ARFF_MEMBER not in archive.namelist():
                raise ValueError(
                    "Dry Bean archive is missing the expected ARFF dataset"
                )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Dry Bean archive is not a valid zip file") from error


def _validated_dry_bean_parts(df: pd.DataFrame):
    if "target" not in df.columns:
        raise ValueError("Processed Dry Bean data is missing the target column")
    if df.empty:
        raise ValueError("Processed Dry Bean data must contain rows")

    X = df.drop(columns=["target"])
    if X.shape[1] == 0 or X.isna().any().any():
        raise ValueError("Dry Bean data must contain complete feature columns")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Dry Bean features must all be numeric")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Dry Bean features must all be finite")

    try:
        target = pd.to_numeric(df["target"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Dry Bean targets must be integer class IDs") from error
    target_values = target.to_numpy(dtype=float)
    if (
        not np.isfinite(target_values).all()
        or not np.equal(target_values, np.floor(target_values)).all()
    ):
        raise ValueError("Dry Bean targets must be finite integer class IDs")

    y = target.astype(int)
    classes = np.sort(y.unique())
    if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Dry Bean targets must be contiguous class IDs starting at zero")
    return X, y


def download_dry_bean(zip_path=DRY_BEAN_ZIP_PATH, url: str = DRY_BEAN_URL) -> Path:
    """Download the UCI Dry Bean zip file if it is not already present."""

    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        temporary_path = None
        try:
            with NamedTemporaryFile(
                prefix=f".{zip_path.name}.",
                suffix=".tmp",
                dir=zip_path.parent,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
            urlretrieve(url, temporary_path)
            _validate_dry_bean_archive(temporary_path)
            temporary_path.replace(zip_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    _validate_dry_bean_archive(zip_path)
    return zip_path


def extract_dry_bean_arff(
    zip_path=DRY_BEAN_ZIP_PATH,
    raw_dir=DRY_BEAN_RAW_DIR,
    arff_member: str = DRY_BEAN_ARFF_MEMBER,
) -> Path:
    """Extract the Dry Bean ARFF file from the downloaded UCI zip."""

    zip_path = Path(zip_path)
    raw_dir = Path(raw_dir)
    arff_path = raw_dir / arff_member

    if not arff_path.exists():
        with ZipFile(zip_path) as archive:
            archive.extract(arff_member, path=raw_dir)

    return arff_path


def parse_dry_bean_arff(arff_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the Dry Bean ARFF file into a numeric dataframe and label mapping."""

    data, _ = arff.loadarff(arff_path)
    df = pd.DataFrame(data)

    if "Class" not in df.columns:
        raise ValueError("Dry Bean ARFF is missing the Class attribute")

    class_labels = df["Class"].str.decode("utf-8")
    if class_labels.isna().any():
        raise ValueError("Dry Bean ARFF contains missing class labels")
    label_encoder = LabelEncoder()
    df = df.drop(columns=["Class"])
    if df.empty or df.isna().any().any():
        raise ValueError("Dry Bean ARFF must contain non-empty, complete feature rows")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        raise ValueError("Dry Bean features must all be numeric")
    df["target"] = label_encoder.fit_transform(class_labels)
    _validated_dry_bean_parts(df)

    mapping = pd.DataFrame(
        {
            "target": range(len(label_encoder.classes_)),
            "class_label": label_encoder.classes_,
        }
    )

    return df, mapping


def prepare_dry_bean_dataset(
    zip_path=DRY_BEAN_ZIP_PATH,
    raw_dir=DRY_BEAN_RAW_DIR,
    processed_path=DRY_BEAN_PROCESSED_PATH,
    label_mapping_path=DRY_BEAN_LABEL_MAPPING_PATH,
    force: bool = False,
) -> Path:
    """Download, parse, and cache the UCI Dry Bean dataset as processed CSV."""

    processed_path = Path(processed_path)
    label_mapping_path = Path(label_mapping_path)

    if processed_path.exists() and label_mapping_path.exists() and not force:
        _validated_dry_bean_parts(load_dataframe(processed_path))
        return processed_path

    zip_path = download_dry_bean(zip_path=zip_path)
    arff_path = extract_dry_bean_arff(zip_path=zip_path, raw_dir=raw_dir)
    df, mapping = parse_dry_bean_arff(arff_path)
    save_dataframe_atomic(mapping, label_mapping_path)
    save_dataframe_atomic(df, processed_path)

    return processed_path


def load_dry_bean(auto_prepare: bool = True) -> DatasetBundle:
    """Load the UCI Dry Bean dataset.

    If the processed CSV is missing, ``auto_prepare=True`` downloads and parses
    the UCI source archive.
    """

    if not DRY_BEAN_PROCESSED_PATH.exists():
        if not auto_prepare:
            raise FileNotFoundError(
                "Dry Bean is not prepared. Run "
                "`python scripts/download_external_datasets.py` first."
            )
        prepare_dry_bean_dataset()

    df = load_dataframe(DRY_BEAN_PROCESSED_PATH)
    X, y = _validated_dry_bean_parts(df)

    return DatasetBundle(
        name="dry_bean",
        X=X,
        y=y,
        task_type="multiclass",
        n_classes=int(y.nunique()),
    )
