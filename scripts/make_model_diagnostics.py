"""Create model-level diagnostic tables and plots for one experiment setting."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_final_models import (
    model_artifact_dir,
    save_final_model_set,
    train_final_model_set,
)
from src.utils.io import save_dataframe
from src.utils.plotting import CONDITION_LABELS, save_report_figure


def _plot_confusion_matrix(model, X_test, y_test, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    save_report_figure(fig, path)
    plt.close(fig)


def _plot_probability_curve(model, X_test, y_test, curve_type, title, path):
    if not hasattr(model, "predict_proba"):
        return None

    y_proba = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    y_test_array = np.asarray(y_test)

    if len(classes) == 2:
        positive_scores = y_proba[:, 1]
        if curve_type == "roc":
            x_values, y_values, _ = roc_curve(y_test_array, positive_scores, pos_label=classes[1])
            score = auc(x_values, y_values)
            xlabel, ylabel = "False positive rate", "True positive rate"
            label = f"ROC AUC = {score:.3f}"
        else:
            y_values, x_values, _ = precision_recall_curve(
                y_test_array,
                positive_scores,
                pos_label=classes[1],
            )
            score = auc(x_values, y_values)
            xlabel, ylabel = "Recall", "Precision"
            label = f"PR AUC = {score:.3f}"
    else:
        y_binary = label_binarize(y_test_array, classes=classes)
        if curve_type == "roc":
            x_values, y_values, _ = roc_curve(y_binary.ravel(), y_proba.ravel())
            score = auc(x_values, y_values)
            xlabel, ylabel = "False positive rate", "True positive rate"
            label = f"Micro ROC AUC = {score:.3f}"
        else:
            y_values, x_values, _ = precision_recall_curve(
                y_binary.ravel(),
                y_proba.ravel(),
            )
            score = auc(x_values, y_values)
            xlabel, ylabel = "Recall", "Precision"
            label = f"Micro PR AUC = {score:.3f}"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_values, y_values, label=label)
    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Chance")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    save_report_figure(fig, path)
    plt.close(fig)

    return score


def _classification_report_frame(model, X_test, y_test, condition):
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    # sklearn's scalar accuracy is not a per-class row (in particular, not support).
    frame = pd.DataFrame({
        label: metrics for label, metrics in report.items() if isinstance(metrics, dict)
    }).T.reset_index().rename(columns={"index": "label"})
    frame.insert(0, "condition", condition)
    return frame


def _plot_f1_summary(summary, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["condition"].map(CONDITION_LABELS), summary["f1_macro"])
    ax.set_title("Macro F1 by training condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save_report_figure(fig, path)
    plt.close(fig)


def create_model_diagnostics(
    dataset_name: str,
    model_name: str,
    noise_type: str,
    noise_rate: float,
    detector: str,
    random_state: int,
    cv_folds: int,
    table_dir,
    figure_dir,
    save_models: bool = False,
    model_output_dir="results/models",
):
    """Train one model set and save diagnostic tables/figures."""

    artifacts = train_final_model_set(
        dataset_name=dataset_name,
        model_name=model_name,
        noise_type=noise_type,
        noise_rate=noise_rate,
        detector=detector,
        random_state=random_state,
        cv_folds=cv_folds,
    )

    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    summary = artifacts["metrics"].copy()
    report_frames = []

    for condition, model in artifacts["models"].items():
        slug = condition
        _plot_confusion_matrix(
            model,
            X_test,
            y_test,
            title=f"{CONDITION_LABELS[condition]}: confusion matrix",
            path=figure_dir / f"{slug}_confusion_matrix.png",
        )
        roc_auc = _plot_probability_curve(
            model,
            X_test,
            y_test,
            curve_type="roc",
            title=f"{CONDITION_LABELS[condition]}: ROC curve",
            path=figure_dir / f"{slug}_roc_curve.png",
        )
        pr_auc = _plot_probability_curve(
            model,
            X_test,
            y_test,
            curve_type="pr",
            title=f"{CONDITION_LABELS[condition]}: precision-recall curve",
            path=figure_dir / f"{slug}_precision_recall_curve.png",
        )

        summary.loc[summary["condition"] == condition, "diagnostic_roc_auc"] = roc_auc
        summary.loc[summary["condition"] == condition, "diagnostic_pr_auc"] = pr_auc
        report_frames.append(_classification_report_frame(model, X_test, y_test, condition))

    _plot_f1_summary(summary, figure_dir / "macro_f1_by_condition.png")

    reports = pd.concat(report_frames, ignore_index=True)
    save_dataframe(summary, table_dir / "diagnostic_summary.csv")
    save_dataframe(reports, table_dir / "classification_report_by_condition.csv")

    model_metadata = None
    if save_models:
        artifact_dir = model_artifact_dir(
            model_output_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            noise_type=noise_type,
            noise_rate=noise_rate,
            detector=detector,
            random_state=random_state,
        )
        model_metadata = save_final_model_set(artifacts, artifact_dir)

    return {
        "summary": summary,
        "classification_reports": reports,
        "artifacts": artifacts,
        "model_metadata": model_metadata,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dry_bean")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--noise-type", default="symmetric")
    parser.add_argument("--noise-rate", type=float, default=0.2)
    parser.add_argument("--detector", default="confidence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--table-dir", default="results/tables/model_diagnostics")
    parser.add_argument("--figure-dir", default="results/figures/model_diagnostics")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--model-output-dir", default="results/models")
    return parser.parse_args()


def main():
    args = parse_args()
    result = create_model_diagnostics(
        dataset_name=args.dataset,
        model_name=args.model,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        detector=args.detector,
        random_state=args.seed,
        cv_folds=args.cv_folds,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        save_models=args.save_models,
        model_output_dir=args.model_output_dir,
    )

    print(f"Saved diagnostic tables to {args.table_dir}")
    print(f"Saved diagnostic figures to {args.figure_dir}")
    if result["model_metadata"] is not None:
        print(f"Saved fitted models to {args.model_output_dir}")
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
