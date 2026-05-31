"""External public tabular datasets used beyond the sklearn smoke datasets."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

from src.data.load_datasets import DatasetBundle
from src.utils.io import save_dataframe


DRY_BEAN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00602/DryBeanDataset.zip"
)
DRY_BEAN_RAW_DIR = Path("datasets/raw/dry_bean")
DRY_BEAN_ZIP_PATH = DRY_BEAN_RAW_DIR / "DryBeanDataset.zip"
DRY_BEAN_ARFF_MEMBER = "DryBeanDataset/Dry_Bean_Dataset.arff"
DRY_BEAN_PROCESSED_PATH = Path("datasets/processed/dry_bean.csv")
DRY_BEAN_LABEL_MAPPING_PATH = Path("datasets/processed/dry_bean_label_mapping.csv")


def download_dry_bean(zip_path=DRY_BEAN_ZIP_PATH, url: str = DRY_BEAN_URL) -> Path:
    """Download the UCI Dry Bean zip file if it is not already present."""

    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        urlretrieve(url, zip_path)

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

    class_labels = df["Class"].str.decode("utf-8")
    label_encoder = LabelEncoder()
    df = df.drop(columns=["Class"])
    df["target"] = label_encoder.fit_transform(class_labels)

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

    if processed_path.exists() and not force:
        return processed_path

    zip_path = download_dry_bean(zip_path=zip_path)
    arff_path = extract_dry_bean_arff(zip_path=zip_path, raw_dir=raw_dir)
    df, mapping = parse_dry_bean_arff(arff_path)
    save_dataframe(df, processed_path)
    save_dataframe(mapping, label_mapping_path)

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

    df = pd.read_csv(DRY_BEAN_PROCESSED_PATH)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    return DatasetBundle(
        name="dry_bean",
        X=X,
        y=y,
        task_type="multiclass",
        n_classes=int(y.nunique()),
    )
