"""Run the complete noisy-label experiment pipeline in order."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_experiment_config, quick_experiment_config


EXTERNAL_DATASETS = {"dry_bean"}


def _python_command(*args):
    return [sys.executable, *args]


def build_pipeline_commands(args):
    """Build ordered subprocess commands for the full project pipeline."""

    config = load_experiment_config(args.config)
    matrix_config = quick_experiment_config(config) if args.quick else config
    commands = []

    if args.quick:
        results_path = args.results or "results/tables/quick_pipeline_results.csv"
        summary_path = args.summary or "results/tables/quick_pipeline_summary.csv"
        analysis_dir = args.analysis_dir or "results/tables/quick_pipeline_analysis"
        figure_dir = args.figure_dir or "results/figures/quick_pipeline"
        metadata_dir = args.metadata_dir or "results/tables/quick_pipeline_metadata"
    else:
        results_path = args.results or config["output"]["results_csv"]
        summary_path = args.summary or config["output"]["summary_csv"]
        analysis_dir = args.analysis_dir or "results/tables/analysis"
        figure_dir = args.figure_dir or "results/figures"
        metadata_dir = args.metadata_dir or "results/tables"

    external_datasets = [
        dataset
        for dataset in matrix_config["datasets"]
        if dataset in EXTERNAL_DATASETS
    ]
    if not args.skip_download and external_datasets:
        commands.append(
            _python_command(
                "scripts/download_external_datasets.py",
                "--datasets",
                *external_datasets,
            )
        )

    if not args.skip_training:
        training_command = _python_command(
            "scripts/run_cleaning_experiments.py",
            "--config",
            args.config,
        )
        if args.quick:
            training_command.append("--quick")
        if args.resume:
            training_command.append("--resume")
        training_command.extend(["--output", results_path])
        commands.append(training_command)

    commands.extend(
        [
            _python_command(
                "scripts/make_summary_tables.py",
                "--input",
                results_path,
                "--output",
                summary_path,
            ),
            _python_command(
                "scripts/make_analysis_tables.py",
                "--input",
                results_path,
                "--output-dir",
                analysis_dir,
            ),
            _python_command(
                "scripts/make_plots.py",
                "--input",
                results_path,
                "--output-dir",
                figure_dir,
            ),
            _python_command(
                "scripts/make_metadata_tables.py",
                "--config",
                args.config,
                "--output-dir",
                metadata_dir,
                *(["--quick"] if args.quick else []),
            ),
        ]
    )

    if not args.quick:
        commands.append(
            _python_command(
                "scripts/validate_results.py",
                "--results",
                results_path,
                "--config",
                args.config,
                "--output",
                args.validation_report,
            )
        )

    manifest_path = args.run_manifest
    if manifest_path is None:
        manifest_path = (
            "results/logs/quick_pipeline_manifest.json"
            if args.quick
            else "results/logs/run_manifest.json"
        )
    manifest_command = _python_command(
        "scripts/make_run_manifest.py",
        "--config",
        args.config,
        "--results",
        results_path,
        "--output",
        manifest_path,
    )
    if args.quick:
        manifest_command.append("--quick")
    commands.append(manifest_command)

    artifact_args = [
        "--dataset",
        args.artifact_dataset,
        "--model",
        args.artifact_model,
        "--noise-type",
        args.artifact_noise_type,
        "--noise-rate",
        str(args.artifact_noise_rate),
        "--detector",
        args.artifact_detector,
        "--seed",
        str(args.artifact_seed),
        "--cv-folds",
        str(args.artifact_cv_folds),
    ]

    if args.save_final_models and not args.make_model_diagnostics:
        commands.append(
            _python_command(
                "scripts/train_final_models.py",
                *artifact_args,
                "--output-dir",
                args.model_output_dir,
            )
        )

    if args.make_model_diagnostics:
        commands.append(
            _python_command(
                "scripts/make_model_diagnostics.py",
                *artifact_args,
                "--table-dir",
                args.diagnostic_table_dir,
                "--figure-dir",
                args.diagnostic_figure_dir,
                *(
                    ["--save-models", "--model-output-dir", args.model_output_dir]
                    if args.save_final_models
                    else []
                ),
            )
        )

    return commands


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Override result CSV path.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Override summary CSV path.",
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="Directory for analysis CSV files.",
    )
    parser.add_argument(
        "--figure-dir",
        default=None,
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--metadata-dir",
        default=None,
        help="Directory for generated experiment metadata tables.",
    )
    parser.add_argument(
        "--validation-report",
        default="results/tables/validation_report.csv",
        help="Output CSV for validation report.",
    )
    parser.add_argument(
        "--run-manifest",
        default=None,
        help="Output JSON path for config, data, result, Git, and environment hashes.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick one-combination training check and skip full-matrix validation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training by skipping completed result combinations.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip external dataset download/preparation.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse an existing result CSV and only regenerate derived outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--save-final-models",
        action="store_true",
        help="Train and save final fitted models for one representative setting.",
    )
    parser.add_argument(
        "--make-model-diagnostics",
        action="store_true",
        help="Generate confusion matrices, ROC curves, PR curves, and class reports.",
    )
    parser.add_argument("--artifact-dataset", default="dry_bean")
    parser.add_argument("--artifact-model", default="random_forest")
    parser.add_argument("--artifact-noise-type", default="symmetric")
    parser.add_argument("--artifact-noise-rate", type=float, default=0.2)
    parser.add_argument("--artifact-detector", default="confidence")
    parser.add_argument("--artifact-seed", type=int, default=42)
    parser.add_argument("--artifact-cv-folds", type=int, default=5)
    parser.add_argument("--model-output-dir", default="results/models")
    parser.add_argument(
        "--diagnostic-table-dir",
        default="results/tables/model_diagnostics",
    )
    parser.add_argument(
        "--diagnostic-figure-dir",
        default="results/figures/model_diagnostics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    commands = build_pipeline_commands(args)

    for command in commands:
        printable = " ".join(command)
        print(f"Running: {printable}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
