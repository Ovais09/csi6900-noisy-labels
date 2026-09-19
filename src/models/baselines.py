"""Baseline model definitions for tabular classification experiments."""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUPPORTED_MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
)


def get_baseline_models(random_state: int = 42):
    """Return the baseline classifiers used throughout the benchmark."""

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=random_state,
        ),
    }
