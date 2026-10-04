"""Certify and freeze the canonical code/results release artifacts."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import joblib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_analysis_tables import make_all_analysis_tables
from scripts.make_metadata_tables import (
    CLEANING_ROWS,
    dataset_table,
    detector_table,
    model_table,
    noise_table,
)
from scripts.make_summary_tables import summarize_results
from scripts.train_final_models import MODEL_CONDITIONS, model_artifact_dir
from scripts.validate_results import validate_results, validation_report
from src.data.load_datasets import load_dataset
from src.data.preprocessing import make_train_val_test_split
from src.evaluation.metrics import classification_metrics
from src.noise.inject_noise import inject_class_dependent_noise, inject_symmetric_noise
from src.utils.config import load_experiment_config
from src.utils.io import load_dataframe, save_json_atomic
from src.utils.provenance import dataset_fingerprint, git_state, sha256_file
from src.utils.release import (
    build_release_manifest,
    load_release_manifest,
    verify_release_manifest,
)
from src.utils.training import training_manifest_path, verify_training_record

ANALYSIS_FILENAMES = (
    "condition_summary.csv",
    "detector_ranking.csv",
    "detector_breakdown.csv",
    "cleaning_gain_summary.csv",
    "noise_impact_summary.csv",
    "best_cleaning_by_setting.csv",
    "paired_cleaning_comparisons.csv",
    "combined_detector_comparisons.csv",
)
METADATA_FILENAMES = (
    "dataset_overview.csv",
    "model_overview.csv",
    "noise_settings.csv",
    "detector_methods.csv",
    "cleaning_conditions.csv",
)
DIAGNOSTIC_TABLE_FILENAMES = (
    "diagnostic_summary.csv",
    "classification_report_by_condition.csv",
)
DIAGNOSTIC_FIGURE_FILENAMES = (
    "clean_upper_bound_confusion_matrix.png",
    "clean_upper_bound_precision_recall_curve.png",
    "clean_upper_bound_roc_curve.png",
    "macro_f1_by_condition.png",
    "noisy_baseline_confusion_matrix.png",
    "noisy_baseline_precision_recall_curve.png",
    "noisy_baseline_roc_curve.png",
    "relabel_suspicious_confusion_matrix.png",
    "relabel_suspicious_precision_recall_curve.png",
    "relabel_suspicious_roc_curve.png",
    "remove_suspicious_confusion_matrix.png",
    "remove_suspicious_precision_recall_curve.png",
    "remove_suspicious_roc_curve.png",
)


@dataclass(frozen=True)
class AuditCheck:
    status: str
    check: str
    detail: str


def _read_csv(path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    frame = load_dataframe(path)
    if frame.empty:
        raise ValueError(f"CSV contains no rows: {path}")
    return frame


def _assert_frame_matches(actual_path, expected: pd.DataFrame) -> pd.DataFrame:
    actual = load_dataframe(actual_path)
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        headline = str(error).splitlines()[0]
        raise ValueError(
            f"Saved table differs from a fresh recomputation: {actual_path} ({headline})"
        ) from error
    return actual


def _validate_png(path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing PNG: {path}")
    image = np.asarray(mpimg.imread(path))
    if image.ndim < 2 or image.shape[0] < 200 or image.shape[1] < 300:
        raise ValueError(f"PNG has invalid dimensions: {path}")
    if not np.isfinite(image).all():
        raise ValueError(f"PNG contains non-finite pixels: {path}")
    spatial_range = (
        np.ptp(image, axis=(0, 1))
        if image.ndim > 2
        else np.ptp(image)
    )
    if float(np.max(spatial_range)) <= 1e-6:
        raise ValueError(f"PNG appears blank: {path}")


def _check_vector_companions(png_paths):
    paths = [path.with_suffix(".pdf") for path in png_paths]
    for path in paths:
        content = path.read_bytes()
        if not content.startswith(b"%PDF-") or not content.rstrip().endswith(b"%%EOF"):
            raise ValueError(f"Missing or malformed vector PDF: {path}")
    return paths


def expected_main_figures(config, figure_dir) -> list[Path]:
    """Return the exact canonical figure set for an experiment config."""

    figure_dir = Path(figure_dir)
    paths = []
    for dataset in config["datasets"]:
        for model in config["models"]:
            paths.append(figure_dir / f"{dataset}_{model}_detection_f1.png")
        for noise_type in config["noise_types"]:
            for model in config["models"]:
                for metric in ("accuracy", "f1_macro"):
                    paths.append(
                        figure_dir
                        / f"{dataset}_{noise_type}_{model}_{metric}.png"
                    )
    return paths


def _check_result_matrix(results, config):
    issues = validate_results(results, config)
    failures = [issue for issue in issues if issue.status == "FAIL"]
    if failures:
        detail = "; ".join(f"{issue.check}: {issue.detail}" for issue in failures)
        raise ValueError(detail)
    return f"{len(issues)} contract checks passed"


def _check_splits_and_noise(results, config):
    """Reconstruct source splits and corruption independently of saved counts."""
    split_count = 0
    for name in config["datasets"]:
        dataset = load_dataset(name)
        if dataset.X.duplicated().any():
            raise ValueError(f"Duplicate feature rows could leak across splits: {name}")
        original_y = dataset.y.copy()
        for seed in config["seeds"]:
            X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(
                dataset.X, dataset.y, random_state=seed,
            )
            indices = [set(frame.index) for frame in (X_train, X_val, X_test)]
            if any(indices[i] & indices[j] for i in range(3) for j in range(i + 1, 3)):
                raise ValueError(f"Overlapping split indices: {name}, seed={seed}")
            if set.union(*indices) != set(dataset.X.index):
                raise ValueError(f"Split coverage differs from source: {name}")
            for noise_type in config["noise_types"]:
                injector = (inject_symmetric_noise if noise_type == "symmetric"
                            else inject_class_dependent_noise)
                for rate in config["noise_rates"]:
                    _, mask = injector(y_train, rate, random_state=seed)
                    rows = results[
                        results["dataset"].eq(name) & results["random_state"].eq(seed)
                        & results["noise_type"].eq(noise_type) & results["noise_rate"].eq(rate)
                    ]
                    expected = {"n_train": len(X_train), "n_test": len(X_test),
                                "n_corrupted": int(mask.sum())}
                    if rows.empty or any(not rows[key].eq(value).all()
                                         for key, value in expected.items()):
                        raise ValueError(f"Reconstructed split/noise counts differ: {name}")
            pd.testing.assert_series_equal(y_val, original_y.loc[y_val.index])
            pd.testing.assert_series_equal(y_test, original_y.loc[y_test.index])
            pd.testing.assert_series_equal(dataset.y, original_y)
            split_count += 1
    return f"{split_count} disjoint splits, untouched holdouts, and exact corruption counts"


def _check_summary(results, summary_path):
    expected = summarize_results(results)
    actual = _assert_frame_matches(summary_path, expected)
    return f"{len(actual)} seed-aggregated rows", [Path(summary_path)]


def _check_analysis(results, analysis_dir):
    analysis_dir = Path(analysis_dir)
    expected_tables = make_all_analysis_tables(results)
    paths = []
    rows = 0
    for filename in ANALYSIS_FILENAMES:
        name = Path(filename).stem
        path = analysis_dir / filename
        actual = _assert_frame_matches(path, expected_tables[name])
        paths.append(path)
        rows += len(actual)
    return f"{len(paths)} exact tables, {rows} total rows", paths


def _check_metadata(config, metadata_dir):
    metadata_dir = Path(metadata_dir)
    expected_tables = {
        "dataset_overview.csv": dataset_table(config["datasets"]),
        "model_overview.csv": model_table(config["models"]),
        "noise_settings.csv": noise_table(
            config["noise_types"],
            config["noise_rates"],
        ),
        "detector_methods.csv": detector_table(config["detectors"]),
        "cleaning_conditions.csv": pd.DataFrame(CLEANING_ROWS),
    }
    paths = []
    for filename in METADATA_FILENAMES:
        path = metadata_dir / filename
        _assert_frame_matches(path, expected_tables[filename])
        paths.append(path)
    return f"{len(paths)} exact metadata tables", paths


def _check_validation_report(results, config, report_path):
    expected = validation_report(validate_results(results, config))
    actual = _assert_frame_matches(report_path, expected)
    if not actual["status"].eq("PASS").all():
        raise ValueError("Saved validation report contains failures")
    return f"{len(actual)} saved checks are PASS", [Path(report_path)]


def _check_main_figures(config, figure_dir):
    paths = expected_main_figures(config, figure_dir)
    expected_names = {path.name for path in paths}
    actual_names = {path.name for path in Path(figure_dir).glob("*.png")}
    if actual_names != expected_names:
        raise ValueError(
            "Canonical figure set differs from config: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for path in paths:
        _validate_png(path)
    return f"{len(paths)} nonblank PNGs with vector PDF companions", [
        *paths, *_check_vector_companions(paths),
    ]


def _resolve_artifact_path(path, project_root):
    path = Path(path)
    return path if path.is_absolute() else Path(project_root) / path


def _check_model_artifacts(
    dataset_name,
    model_name,
    noise_type,
    noise_rate,
    detector,
    random_state,
    model_output_dir,
    project_root,
    results=None,
    cv_folds=None,
):
    artifact_dir = model_artifact_dir(
        model_output_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        noise_type=noise_type,
        noise_rate=noise_rate,
        detector=detector,
        random_state=random_state,
    )
    metadata_path = artifact_dir / "model_metadata.csv"
    metadata = _read_csv(metadata_path)
    if set(metadata["condition"]) != set(MODEL_CONDITIONS) or len(metadata) != 4:
        raise ValueError("Model metadata must contain each comparison condition once")

    selectors = {
        "dataset": dataset_name,
        "model": model_name,
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "random_state": random_state,
    }
    for column, expected in selectors.items():
        if not metadata[column].eq(expected).all():
            raise ValueError(f"Model metadata has an unexpected {column}")
    expected_detectors = metadata["condition"].map({
        "clean_upper_bound": "none", "noisy_baseline": "none",
        "remove_suspicious": detector, "relabel_suspicious": detector,
    })
    if not metadata["detector"].eq(expected_detectors).all():
        raise ValueError("Model metadata has an unexpected detector")
    if cv_folds is not None and not metadata["cv_folds"].eq(cv_folds).all():
        raise ValueError("Model metadata has an unexpected cv_folds")
    if results is not None:
        subset = results.copy()
        for column, expected in selectors.items():
            subset = subset[subset[column].eq(expected)]
        subset = subset[subset["detector"].isin(["none", detector])]
        try:
            pd.testing.assert_frame_equal(
                metadata[list(results.columns)].sort_values("condition").reset_index(drop=True),
                subset.sort_values("condition").reset_index(drop=True),
                check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
            )
        except AssertionError as error:
            raise ValueError("Saved models disagree with the canonical result matrix") from error

    dataset = load_dataset(dataset_name)
    _, _, X_test, _, _, y_test = make_train_val_test_split(
        dataset.X,
        dataset.y,
        random_state=random_state,
    )
    paths = [metadata_path]
    for row in metadata.itertuples(index=False):
        path = _resolve_artifact_path(row.model_path, project_root)
        if not path.is_file():
            raise FileNotFoundError(f"Missing fitted model: {path}")
        if path.stat().st_size != int(row.model_bytes):
            raise ValueError(f"Saved model size mismatch: {path}")
        if sha256_file(path) != row.model_sha256:
            raise ValueError(f"Saved model hash mismatch: {path}")
        model = joblib.load(path)
        if not hasattr(model, "predict"):
            raise ValueError(f"Saved artifact is not a classifier: {path}")
        metrics = classification_metrics(model, X_test, y_test)
        for metric in ("accuracy", "f1_macro", "auroc"):
            if not np.isclose(metrics[metric], getattr(row, metric), rtol=1e-12, atol=1e-12):
                raise ValueError(f"Reloaded model {metric} mismatch: {path}")
        paths.append(path)

    return "4 models reloaded, hashed, and reproduced their metrics", paths, metadata


def _check_diagnostics(table_dir, figure_dir, model_metadata, project_root=PROJECT_ROOT):
    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)
    summary_path = table_dir / "diagnostic_summary.csv"
    report_path = table_dir / "classification_report_by_condition.csv"
    summary = _read_csv(summary_path)
    report = _read_csv(report_path)

    if set(summary["condition"]) != set(MODEL_CONDITIONS) or len(summary) != 4:
        raise ValueError("Diagnostic summary has an invalid condition set")
    if set(report["condition"]) != set(MODEL_CONDITIONS):
        raise ValueError("Classification report has an invalid condition set")
    metadata_columns = [column for column in model_metadata
                        if column not in {"model_path", "model_bytes", "model_sha256"}]
    pd.testing.assert_frame_equal(
        summary[metadata_columns], model_metadata[metadata_columns],
        check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
    )
    merged = summary.merge(
        model_metadata[
            ["condition", "accuracy", "f1_macro", "auroc"]
        ],
        on="condition",
        suffixes=("_diagnostic", "_model"),
        validate="one_to_one",
    )
    for metric in ("accuracy", "f1_macro", "auroc"):
        if not np.allclose(
            merged[f"{metric}_diagnostic"],
            merged[f"{metric}_model"],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(f"Diagnostic and model metadata disagree on {metric}")
    for metric in ("diagnostic_roc_auc", "diagnostic_pr_auc"):
        if not summary[metric].between(0.0, 1.0).all():
            raise ValueError(f"Diagnostic metric is outside [0, 1]: {metric}")

    first = model_metadata.iloc[0]
    dataset = load_dataset(first["dataset"])
    _, _, X_test, _, _, y_test = make_train_val_test_split(
        dataset.X, dataset.y, random_state=int(first["random_state"]),
    )
    expected_reports = []
    for row in model_metadata.itertuples(index=False):
        model = joblib.load(_resolve_artifact_path(row.model_path, project_root))
        class_report = classification_report(y_test, model.predict(X_test),
                                             output_dict=True, zero_division=0)
        for label, values in class_report.items():
            if isinstance(values, dict):
                expected_reports.append({"condition": row.condition, "label": label, **values})
        proba = model.predict_proba(X_test)
        binary = label_binarize(y_test, classes=model.classes_)
        targets, scores = ((binary.ravel(), proba[:, 1]) if len(model.classes_) == 2
                           else (binary.ravel(), proba.ravel()))
        precision, recall, _ = precision_recall_curve(targets, scores)
        expected_scores = {"diagnostic_roc_auc": roc_auc_score(targets, scores),
                           "diagnostic_pr_auc": auc(recall, precision)}
        diagnostic = summary.loc[summary["condition"].eq(row.condition)].iloc[0]
        for metric, value in expected_scores.items():
            if not np.isclose(diagnostic[metric], value, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Recomputed diagnostic differs: {row.condition}, {metric}")
    _assert_frame_matches(report_path, pd.DataFrame(expected_reports))

    figure_paths = [figure_dir / name for name in DIAGNOSTIC_FIGURE_FILENAMES]
    actual_names = {path.name for path in figure_dir.glob("*.png")}
    expected_names = set(DIAGNOSTIC_FIGURE_FILENAMES)
    if actual_names != expected_names:
        raise ValueError(
            "Diagnostic figure set is incomplete or contains stale files: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for path in figure_paths:
        _validate_png(path)

    paths = [summary_path, report_path, *figure_paths, *_check_vector_companions(figure_paths)]
    return f"2 tables and {len(figure_paths)} figures are consistent", paths


def _check_processed_datasets(config, processed_dir, project_root):
    processed_dir = Path(processed_dir)
    manifest_path = processed_dir / "manifest.csv"
    manifest = _read_csv(manifest_path)
    required = {
        "dataset",
        "path",
        "samples",
        "features",
        "classes",
        "task_type",
        "sha256",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Dataset export manifest is missing columns: {sorted(missing)}")
    if set(manifest["dataset"]) != set(config["datasets"]):
        raise ValueError("Dataset export manifest does not match the experiment config")

    paths = [manifest_path]
    for row in manifest.itertuples(index=False):
        path = _resolve_artifact_path(row.path, project_root)
        if sha256_file(path) != row.sha256:
            raise ValueError(f"Exported dataset hash mismatch: {path}")
        exported = _read_csv(path)
        dataset = load_dataset(row.dataset)
        expected = dataset.X.copy()
        expected["target"] = dataset.y.to_numpy()
        pd.testing.assert_frame_equal(
            exported,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        if (
            int(row.samples) != len(dataset.y)
            or int(row.features) != dataset.X.shape[1]
            or int(row.classes) != dataset.n_classes
        ):
            raise ValueError(f"Dataset dimensions disagree with manifest: {row.dataset}")
        paths.append(path)
    if "dry_bean" in config["datasets"]:
        from src.data.external_datasets import (
            DRY_BEAN_ARFF_MEMBER,
            DRY_BEAN_LABEL_MAPPING_PATH,
            DRY_BEAN_PROCESSED_PATH,
            DRY_BEAN_RAW_DIR,
            DRY_BEAN_ZIP_PATH,
            parse_dry_bean_arff,
        )
        arff_path = DRY_BEAN_RAW_DIR / DRY_BEAN_ARFF_MEMBER
        with ZipFile(DRY_BEAN_ZIP_PATH) as archive:
            if archive.read(DRY_BEAN_ARFF_MEMBER) != arff_path.read_bytes():
                raise ValueError("Extracted Dry Bean source differs from its raw archive")
        source, mapping = parse_dry_bean_arff(arff_path)
        _assert_frame_matches(DRY_BEAN_PROCESSED_PATH, source)
        _assert_frame_matches(DRY_BEAN_LABEL_MAPPING_PATH, mapping)
        paths.extend([DRY_BEAN_ZIP_PATH, arff_path, DRY_BEAN_PROCESSED_PATH,
                      DRY_BEAN_LABEL_MAPPING_PATH])
    return f"{len(config['datasets'])} exported datasets match their loaders", paths


def _check_run_manifest(manifest_path, config_path, results_path, config, project_root):
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("Run manifest schema is unsupported")
    if manifest.get("configuration", {}).get("sha256") != sha256_file(config_path):
        raise ValueError("Run manifest config hash is stale")
    if manifest.get("configuration", {}).get("values") != config:
        raise ValueError("Run manifest config values are stale")
    if manifest.get("results", {}).get("sha256") != sha256_file(results_path):
        raise ValueError("Run manifest result hash is stale")
    if manifest.get("results", {}).get("rows") != len(_read_csv(results_path)):
        raise ValueError("Run manifest result row count is stale")
    expected_datasets = [dataset_fingerprint(name) for name in config["datasets"]]
    if manifest.get("datasets") != expected_datasets:
        raise ValueError("Run manifest dataset fingerprints are stale")
    current_git = git_state(project_root)
    if not current_git.get("commit"):
        raise ValueError(
            "A final release requires a known source revision; build Docker with "
            "--build-arg VCS_REF=$(git rev-parse HEAD)"
        )
    if manifest.get("git", {}).get("commit") != current_git.get("commit"):
        raise ValueError("Run manifest Git commit is stale")
    if current_git.get("dirty") is True or manifest.get("git", {}).get("dirty") is True:
        raise ValueError("Final release must be audited from a clean Git worktree")
    dependencies = manifest.get("runtime", {}).get("dependencies", {})
    if not dependencies or any(value is None for value in dependencies.values()):
        raise ValueError("Run manifest has incomplete dependency versions")
    training = verify_training_record(config, results_path, project_root)
    if not training.get("git", {}).get("commit"):
        raise ValueError("A final release requires a recorded original training revision")
    if manifest.get("training") != training:
        raise ValueError("Run manifest has stale or missing original training provenance")
    return "config, data, result, environment, and Git fingerprints match", [
        Path(config_path),
        Path(results_path),
        manifest_path,
        training_manifest_path(results_path),
    ]


def _run_check(checks, name, function):
    try:
        result = function()
        if isinstance(result, tuple):
            detail, paths = result
        else:
            detail, paths = result, []
        checks.append(AuditCheck("PASS", name, detail))
        return list(paths)
    except Exception as error:  # noqa: BLE001 - audit boundaries report every failure
        checks.append(AuditCheck("FAIL", name, f"{type(error).__name__}: {error}"))
        return []


def audit_release(args):
    """Run all final artifact checks and optionally freeze their checksums."""

    config = load_experiment_config(args.config)
    results = _read_csv(args.results)
    checks = []
    artifact_paths = [Path(args.results)]

    artifact_paths += _run_check(
        checks,
        "result_matrix",
        lambda: _check_result_matrix(results, config),
    )
    artifact_paths += _run_check(
        checks, "splits_and_corruption", lambda: _check_splits_and_noise(results, config),
    )
    artifact_paths += _run_check(
        checks,
        "summary_table",
        lambda: _check_summary(results, args.summary),
    )
    artifact_paths += _run_check(
        checks,
        "analysis_tables",
        lambda: _check_analysis(results, args.analysis_dir),
    )
    artifact_paths += _run_check(
        checks,
        "metadata_tables",
        lambda: _check_metadata(config, args.metadata_dir),
    )
    artifact_paths += _run_check(
        checks,
        "validation_report",
        lambda: _check_validation_report(results, config, args.validation_report),
    )
    artifact_paths += _run_check(
        checks,
        "main_figures",
        lambda: _check_main_figures(config, args.figure_dir),
    )
    artifact_paths += _run_check(
        checks,
        "processed_datasets",
        lambda: _check_processed_datasets(config, args.processed_dir, PROJECT_ROOT),
    )

    model_result = {}

    def check_models():
        detail, paths, metadata = _check_model_artifacts(
            dataset_name=args.artifact_dataset,
            model_name=args.artifact_model,
            noise_type=args.artifact_noise_type,
            noise_rate=args.artifact_noise_rate,
            detector=args.artifact_detector,
            random_state=args.artifact_seed,
            model_output_dir=args.model_output_dir,
            project_root=PROJECT_ROOT,
            results=results,
            cv_folds=config["cv_folds"],
        )
        model_result["metadata"] = metadata
        return detail, paths

    artifact_paths += _run_check(checks, "saved_models", check_models)
    if "metadata" in model_result:
        artifact_paths += _run_check(
            checks,
            "model_diagnostics",
            lambda: _check_diagnostics(
                args.diagnostic_table_dir,
                args.diagnostic_figure_dir,
                model_result["metadata"],
            ),
        )
    else:
        checks.append(
            AuditCheck(
                "FAIL",
                "model_diagnostics",
                "Saved model verification failed, so diagnostics were not trusted",
            )
        )

    artifact_paths += _run_check(
        checks,
        "run_manifest",
        lambda: _check_run_manifest(
            args.run_manifest,
            args.config,
            args.results,
            config,
            PROJECT_ROOT,
        ),
    )

    release_path = Path(args.release_manifest)
    if not any(check.status == "FAIL" for check in checks):
        try:
            metadata = {
                "config_sha256": sha256_file(args.config),
                "results_sha256": sha256_file(args.results),
                "result_rows": len(results),
                "experiment_combinations": int(
                    len(config["datasets"])
                    * len(config["noise_types"])
                    * len(config["noise_rates"])
                    * len(config["models"])
                    * len(config["seeds"])
                ),
            }
            if args.freeze:
                manifest = build_release_manifest(
                    artifact_paths,
                    project_root=PROJECT_ROOT,
                    metadata=metadata,
                )
                save_json_atomic(manifest, release_path)
            else:
                manifest = load_release_manifest(release_path)
            records = verify_release_manifest(
                manifest,
                project_root=PROJECT_ROOT,
                expected_paths=artifact_paths,
            )
            checks.append(
                AuditCheck(
                    "PASS",
                    "frozen_inventory",
                    f"{len(records)} artifacts match size and SHA-256",
                )
            )
        except Exception as error:  # noqa: BLE001 - audit boundaries report every failure
            checks.append(
                AuditCheck(
                    "FAIL",
                    "frozen_inventory",
                    f"{type(error).__name__}: {error}",
                )
            )
    else:
        checks.append(
            AuditCheck(
                "FAIL",
                "frozen_inventory",
                "Canonical checks failed, so the release was not frozen",
            )
        )

    status = "PASS" if all(check.status == "PASS" for check in checks) else "FAIL"
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": [asdict(check) for check in checks],
        "release_manifest": str(release_path),
        "release_manifest_sha256": (
            sha256_file(release_path) if release_path.is_file() else None
        ),
    }
    save_json_atomic(report, args.output)
    return report


def _artifact_slug(args) -> str:
    rate = f"{args.artifact_noise_rate:g}".replace(".", "p")
    return (
        f"{args.artifact_dataset}_{args.artifact_model}_"
        f"{args.artifact_noise_type}_{rate}_{args.artifact_detector}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--results", default="results/tables/main_experiment_results.csv")
    parser.add_argument("--summary", default="results/tables/summary_results.csv")
    parser.add_argument("--analysis-dir", default="results/tables/analysis")
    parser.add_argument("--metadata-dir", default="results/tables")
    parser.add_argument("--figure-dir", default="results/figures")
    parser.add_argument(
        "--validation-report",
        default="results/tables/validation_report.csv",
    )
    parser.add_argument("--processed-dir", default="datasets/processed")
    parser.add_argument("--run-manifest", default="results/logs/run_manifest.json")
    parser.add_argument("--release-manifest", default="results/logs/release_manifest.json")
    parser.add_argument("--output", default="results/logs/final_audit.json")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--artifact-dataset", default="dry_bean")
    parser.add_argument("--artifact-model", default="random_forest")
    parser.add_argument("--artifact-noise-type", default="symmetric")
    parser.add_argument("--artifact-noise-rate", type=float, default=0.2)
    parser.add_argument("--artifact-detector", default="confidence")
    parser.add_argument("--artifact-seed", type=int, default=42)
    parser.add_argument("--model-output-dir", default="results/models")
    parser.add_argument("--diagnostic-table-dir", default=None)
    parser.add_argument("--diagnostic-figure-dir", default=None)
    args = parser.parse_args()
    slug = _artifact_slug(args)
    if args.diagnostic_table_dir is None:
        args.diagnostic_table_dir = f"results/tables/model_diagnostics/{slug}"
    if args.diagnostic_figure_dir is None:
        args.diagnostic_figure_dir = f"results/figures/model_diagnostics/{slug}"
    return args


def main():
    args = parse_args()
    report = audit_release(args)
    frame = pd.DataFrame(report["checks"])
    print(frame.to_string(index=False))
    print(f"Saved final audit to {args.output}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
