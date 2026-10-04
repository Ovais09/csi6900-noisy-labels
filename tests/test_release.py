from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scripts import make_plots
from scripts.audit_release import (
    _check_splits_and_noise,
    _check_summary,
    _validate_png,
    expected_main_figures,
)
from scripts.make_summary_tables import summarize_results
from src.evaluation.experiment_runner import run_single_experiment
from src.utils.release import build_release_manifest, verify_release_manifest


def _small_results():
    return run_single_experiment(
        dataset_name="breast_cancer",
        noise_type="symmetric",
        noise_rate=0.1,
        model_name="logistic_regression",
        random_state=42,
        cv_folds=3,
        detector_methods=("confidence",),
    )


def test_release_manifest_round_trip_and_detects_same_size_corruption(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("bravo", encoding="utf-8")
    paths = [first, second, first]

    manifest = build_release_manifest(paths, project_root=tmp_path)
    records = verify_release_manifest(
        manifest,
        project_root=tmp_path,
        expected_paths=[first, second],
    )

    assert manifest["artifact_count"] == 2
    assert [record["path"] for record in records] == ["first.txt", "second.txt"]

    second.write_text("wrong", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        verify_release_manifest(manifest, project_root=tmp_path)


def test_release_manifest_rejects_missing_and_outside_artifacts(tmp_path):
    present = tmp_path / "present.txt"
    present.write_text("present", encoding="utf-8")
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="missing"):
        build_release_manifest([missing], project_root=tmp_path)

    outside = tmp_path.parent / "outside-release-artifact.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside the project root"):
            build_release_manifest([outside], project_root=tmp_path)
    finally:
        outside.unlink()


def test_release_manifest_requires_exact_canonical_artifact_set(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    manifest = build_release_manifest([first], project_root=tmp_path)

    with pytest.raises(ValueError, match="canonical set"):
        verify_release_manifest(
            manifest,
            project_root=tmp_path,
            expected_paths=[first, second],
        )


def test_summary_audit_recomputes_values_and_rejects_stale_output(tmp_path):
    results = _small_results()
    summary_path = tmp_path / "summary.csv"
    summary = summarize_results(results)
    summary.to_csv(summary_path, index=False)

    detail, paths = _check_summary(results, summary_path)

    assert detail.endswith("seed-aggregated rows")
    assert paths == [summary_path]

    stale = pd.read_csv(summary_path)
    stale.loc[0, "accuracy_mean"] = 0.0
    stale.to_csv(summary_path, index=False)
    with pytest.raises(ValueError, match="fresh recomputation"):
        _check_summary(results, summary_path)


def test_png_audit_rejects_blank_images(tmp_path):
    valid_path = tmp_path / "valid.png"
    blank_path = tmp_path / "blank.png"

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    fig.savefig(valid_path, dpi=100)
    plt.close(fig)
    plt.imsave(blank_path, [[0.5] * 300] * 200, cmap="gray")

    _validate_png(valid_path)
    with pytest.raises(ValueError, match="blank"):
        _validate_png(blank_path)


def test_expected_main_figure_count_follows_matrix(tmp_path):
    config = {
        "datasets": ["breast_cancer", "wine"],
        "noise_types": ["symmetric", "class_dependent"],
        "models": ["logistic_regression", "random_forest"],
    }

    paths = expected_main_figures(config, tmp_path)

    assert len(paths) == 20
    assert len(set(paths)) == 20
    assert all(Path(path).parent == tmp_path for path in paths)


def test_split_audit_reconstructs_counts_instead_of_trusting_csv():
    results = _small_results()
    config = {"datasets": ["breast_cancer"], "seeds": [42],
              "noise_types": ["symmetric"], "noise_rates": [0.1]}
    assert "disjoint splits" in _check_splits_and_noise(results, config)
    results["n_test"] += 1
    with pytest.raises(ValueError, match="Reconstructed split/noise counts"):
        _check_splits_and_noise(results, config)


@pytest.mark.parametrize("fallback", [False, True])
def test_report_plots_export_nonblank_raster_and_vector_files(tmp_path, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(make_plots, "sns", None)
    results = make_plots.add_plot_labels(_small_results())
    make_plots.plot_metric(results, "accuracy", "breast_cancer", "symmetric",
                           "logistic_regression", tmp_path)
    make_plots.plot_detection_f1(results, "breast_cancer", "logistic_regression", tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 2
    for path in tmp_path.glob("*.png"):
        _validate_png(path)
        assert path.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
