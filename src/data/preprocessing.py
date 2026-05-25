"""Preprocessing and data split helpers."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_numeric_preprocessor():
    """Return the numeric preprocessor used by linear baseline models."""

    return StandardScaler()


def make_train_val_test_split(X, y, random_state: int = 42):
    """Create stratified 60/20/20 train, validation, and test splits."""

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=random_state,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
