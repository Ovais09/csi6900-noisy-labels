"""Derive LaTeX tables and compact figures from the immutable audited release."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_analysis_tables import BASE_COLUMNS, make_all_analysis_tables
from src.utils.io import load_dataframe
from src.utils.provenance import sha256_file

RELEASE_SHA = "26cf086c9804010fc91c86686673d77b498266ebc20e74d47c6fe6df48ecc873"
RESULTS_SHA = "2576804e3caadd3025ff95a9835729ced219fd38dce4ca30ee4a399727a61150"
DATASETS = {"breast_cancer": "Breast Cancer", "wine": "Wine", "digits": "Digits", "dry_bean": "Dry Bean"}
MODELS = {"logistic_regression": "LR", "random_forest": "RF", "hist_gradient_boosting": "HGB"}
DETECTORS = ("confidence", "combined", "disagreement", "margin")
CONDITIONS = (("clean_upper_bound", "none"), ("noisy_baseline", "none")) + tuple(
    (action, detector) for detector in DETECTORS
    for action in ("remove_suspicious", "relabel_suspicious")
)
DIAGNOSTIC = "dry_bean_random_forest_symmetric_0p2_confidence"


def condition_label(condition, detector):
    if condition == "clean_upper_bound":
        return "Clean reference"
    if condition == "noisy_baseline":
        return "Noisy baseline"
    action = "Remove" if condition == "remove_suspicious" else "Relabel"
    return f"{detector.title()} / {action}"


def latex_table(headers, rows, alignment=None):
    """Format already escaped display values in a compact booktabs table."""
    alignment = alignment or "l" + "r" * (len(headers) - 1)
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def paired_gains(frame):
    baseline = frame[frame.condition.eq("noisy_baseline")][BASE_COLUMNS + ["accuracy"]]
    cleaned = frame[frame.condition.isin(["remove_suspicious", "relabel_suspicious"])]
    joined = cleaned.merge(baseline, on=BASE_COLUMNS, validate="many_to_one", suffixes=("", "_noisy"))
    joined["gain_pp"] = 100 * (joined.accuracy - joined.accuracy_noisy)
    return joined


def build_report_assets(release_root, output):
    root, output = Path(release_root).resolve(), Path(output).resolve()
    manifest_path = root / "results/logs/release_manifest.json"
    if sha256_file(manifest_path) != RELEASE_SHA:
        raise ValueError("Report requires the immutable 50f8995 release manifest")
    manifest = json.loads(manifest_path.read_text())
    inventory = {record["path"]: record for record in manifest["artifacts"]}
    inputs = []

    def verified(name):
        path = root / name
        record = inventory[name]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Frozen report input changed: {name}")
        inputs.append(record)
        return path

    results_path = verified("results/tables/main_experiment_results.csv")
    if sha256_file(results_path) != RESULTS_SHA:
        raise ValueError("Unexpected report result matrix")
    df = load_dataframe(results_path)
    tables = make_all_analysis_tables(df)
    for name, expected in tables.items():
        saved = load_dataframe(verified(f"results/tables/analysis/{name}.csv"))
        pd.testing.assert_frame_equal(saved, expected.reset_index(drop=True), check_dtype=False, rtol=1e-10, atol=1e-12)
    datasets = load_dataframe(verified("results/tables/dataset_overview.csv"))
    diagnostic = load_dataframe(verified(f"results/tables/model_diagnostics/{DIAGNOSTIC}/diagnostic_summary.csv"))
    output.mkdir(parents=True, exist_ok=True)

    def write(name, headers, rows, alignment=None):
        (output / f"{name}.tex").write_text(latex_table(headers, rows, alignment), encoding="utf-8")

    aggregate = df.groupby(["condition", "detector"])[["accuracy", "f1_macro", "auroc"]].mean()
    baseline = aggregate.loc[("noisy_baseline", "none"), "accuracy"]
    rows = []
    for condition, detector in CONDITIONS:
        row = aggregate.loc[(condition, detector)]
        rows.append([condition_label(condition, detector), *[f"{row[m]:.4f}" for m in row.index],
                     f"{100 * (row.accuracy - baseline):+.2f}"])
    write("overall", ["Method", "Accuracy", "Macro F1", "AUROC", r"$\Delta$ acc. (pp)"], rows)
    aggregate.to_csv(output / "overall.csv")

    rows = []
    for row in datasets.itertuples():
        subset = df[df.dataset.eq(row.dataset)]
        n_train, n_test = int(subset.n_train.max()), int(subset.n_test.iloc[0])
        rows.append([DATASETS[row.dataset], f"{row.samples:,}", row.features, row.classes,
                     f"{n_train:,}", f"{row.samples - n_train - n_test:,}", f"{n_test:,}"])
    write("datasets", ["Dataset", "$N$", "$p$", "$C$", "Train", "Val.", "Test"], rows)

    rows = []
    for dataset, label in DATASETS.items():
        for model, model_label in MODELS.items():
            group = df[df.dataset.eq(dataset) & df.model.eq(model)].groupby(["condition", "detector"]).accuracy.mean()
            methods = [("clean_upper_bound", "none"), ("noisy_baseline", "none"),
                       ("remove_suspicious", "confidence"), ("relabel_suspicious", "confidence"),
                       ("remove_suspicious", "combined")]
            rows.append([label, model_label, *[f"{group.loc[key]:.4f}" for key in methods]])
    write("models", ["Dataset", "Model", "Clean", "Noisy", "Conf. rem.", "Conf. rel.", "Comb. rem."], rows, "llrrrrr")

    rows = [[r.detector.title(), f"{r.det_precision_mean:.4f}", f"{r.det_recall_mean:.4f}", f"{r.det_f1_mean:.4f}"]
            for r in tables["detector_ranking"].itertuples()]
    write("detection", ["Detector", "Precision", "Recall", "F1"], rows)
    paired = tables["paired_cleaning_comparisons"]
    infer = paired[paired.scope.eq("all") & paired.metric.eq("accuracy")]
    rows = [[condition_label(r.condition, r.detector), f"{100 * r.mean_gain:+.2f}",
             f"[{100 * r.ci95_low:+.2f}, {100 * r.ci95_high:+.2f}]",
             f"{100 * r.win_rate:.1f}", f"{r.paired_permutation_p:.3f}", f"{r.paired_permutation_p_holm:.3f}"]
            for r in infer.itertuples()]
    write("inference", ["Method", "Gain (pp)", r"Exploratory 95\% CI", r"Win \%", "$p$", "$p_H$"], rows)

    comparisons = tables["combined_detector_comparisons"]
    comparisons = comparisons[comparisons.scope.eq("all") & comparisons.metric.eq("accuracy")]
    rows = [[r.reference.title(), "Remove" if r.condition == "remove_suspicious" else "Relabel",
             f"{100 * r.mean_gain:+.2f}", f"{r.paired_permutation_p:.3f}", f"{r.paired_permutation_p_holm:.3f}"]
            for r in comparisons.itertuples()]
    write("ablation", ["Reference detector", "Action", "Combined gain (pp)", "$p$", "$p_H$"], rows, "llrrr")
    rows = [[condition_label(r.condition, r.detector), f"{r.accuracy:.4f}", f"{r.f1_macro:.4f}",
             f"{r.auroc:.4f}", f"{r.diagnostic_pr_auc:.4f}"] for r in diagnostic.itertuples()]
    write("diagnostic", ["Condition", "Accuracy", "Macro F1", "Macro AUROC", "Micro PR AUC"], rows)

    impact = tables["noise_impact_summary"]
    rows = []
    for dataset, label in DATASETS.items():
        subset = impact[impact.dataset.eq(dataset)]
        for noise in ("symmetric", "class_dependent"):
            part = subset[subset.noise_type.eq(noise)].groupby("noise_rate").accuracy_drop_mean.mean()
            rows.append([label, "Symmetric" if noise == "symmetric" else "Cyclic",
                         *[f"{100 * part.loc[rate]:.2f}" for rate in (.1, .2, .3, .4)]])
    write("noise_loss", ["Dataset", "Noise", r"10\%", r"20\%", r"30\%", r"40\%"], rows, "llrrrr")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight"})
    styles = [("clean_upper_bound", "none", "#444444", "--", "o", "Clean"),
              ("noisy_baseline", "none", "#c44536", "-", "s", "Noisy"),
              ("remove_suspicious", "confidence", "#087e8b", "-", "^", "Confidence remove"),
              ("remove_suspicious", "combined", "#75507b", "-.", "D", "Combined remove"),
              ("relabel_suspicious", "confidence", "#527a31", ":", "v", "Confidence relabel")]
    for metric, noise in (("accuracy", "symmetric"), ("accuracy", "class_dependent"), ("f1_macro", "class_dependent")):
        fig, axes = plt.subplots(2, 2, figsize=(7, 5.7), sharex=True, sharey=True)
        for ax, (dataset, label) in zip(axes.flat, DATASETS.items()):
            subset = df[df.dataset.eq(dataset) & df.noise_type.eq(noise)]
            for condition, detector, color, line, marker, legend in styles:
                values = subset[subset.condition.eq(condition) & subset.detector.eq(detector)].groupby("noise_rate")[metric].mean()
                ax.plot(values.index * 100, values, color=color, linestyle=line, marker=marker, markersize=4, label=legend)
            ax.set_title(label, fontsize=11)
            ax.set_xticks([10, 20, 30, 40])
            ax.set_ylim(.5, 1.01)
            ax.grid(alpha=.2)
        for ax in axes[1]:
            ax.set_xlabel("Injected noise (%)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Accuracy" if metric == "accuracy" else "Macro F1")
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9)
        fig.tight_layout(rect=(0, .1, 1, 1))
        fig.savefig(output / f"{metric}_{noise}.pdf", metadata={"CreationDate": None})
        plt.close(fig)

    detection = df[df.condition.eq("remove_suspicious")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharey=True)
    for ax, noise in zip(axes, ("symmetric", "class_dependent")):
        for detector, color, marker in zip(DETECTORS, ("#087e8b", "#75507b", "#527a31", "#c44536"), ("o", "s", "^", "D")):
            values = detection[detection.noise_type.eq(noise) & detection.detector.eq(detector)].groupby("noise_rate").det_f1.mean()
            ax.plot(values.index * 100, values, label=detector.title(), color=color, marker=marker)
        ax.set_title("Symmetric" if noise == "symmetric" else "Cyclic class-dependent")
        ax.set_xlabel("Injected noise (%)")
        ax.set_xticks([10, 20, 30, 40])
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Detection F1")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(output / "detection.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    gains = paired_gains(df)
    heat = gains.groupby(["condition", "detector", "dataset"]).gain_pp.mean().unstack("dataset")
    keys = [key for key in CONDITIONS if key[1] != "none"]
    heat = heat.loc[keys, list(DATASETS)]
    heat.to_csv(output / "dataset_gains.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    scale = np.abs(heat.to_numpy()).max()
    artist = ax.imshow(heat, cmap="RdBu", vmin=-scale, vmax=scale, aspect="auto")
    ax.set_xticks(range(4), list(DATASETS.values()))
    ax.set_yticks(range(8), [condition_label(*key) for key in keys])
    for i in range(8):
        for j in range(4):
            value = heat.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", color="white" if abs(value) > .6 * scale else "black")
    fig.colorbar(artist, ax=ax, label="Accuracy gain (percentage points)")
    fig.tight_layout()
    fig.savefig(output / "dataset_gains.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    for stem in ("noisy_baseline_confusion_matrix", "noisy_baseline_roc_curve", "noisy_baseline_precision_recall_curve"):
        source = verified(f"results/figures/model_diagnostics/{DIAGNOSTIC}/{stem}.pdf")
        shutil.copyfile(source, output / f"{stem}.pdf")

    facts = {"CleanAccuracy": f"{aggregate.loc[('clean_upper_bound', 'none'), 'accuracy']:.4f}",
             "NoisyAccuracy": f"{baseline:.4f}",
             "ConfidenceRemoveGain": f"{100 * (aggregate.loc[('remove_suspicious', 'confidence'), 'accuracy'] - baseline):.2f}",
             "CombinedRemoveGain": f"{100 * (aggregate.loc[('remove_suspicious', 'combined'), 'accuracy'] - baseline):.2f}"}
    (output / "facts.tex").write_text("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in facts.items()) + "\n")
    evidence = {"source_revision": "50f8995a014f4d68714cba7a4ea81e911c859374",
                "result_rows": len(df), "results_sha256": RESULTS_SHA, "facts": facts,
                "inputs": sorted(inputs, key=lambda record: record["path"])}
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", default=".")
    parser.add_argument("--output", default="report/generated")
    args = parser.parse_args()
    evidence = build_report_assets(args.release_root, args.output)
    print(f"Generated report assets from {evidence['result_rows']} verified rows")


if __name__ == "__main__":
    main()
