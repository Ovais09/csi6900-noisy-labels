"""Run the complete noisy-label experiment pipeline in order."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _python_command(*args):
    return [sys.executable, *args]


def build_pipeline_commands(args):
    """Build ordered subprocess commands for the full project pipeline."""

    commands = []

    if not args.skip_download:
        commands.append(_python_command("scripts/download_external_datasets.py"))

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
        if args.results:
            training_command.extend(["--output", args.results])
        commands.append(training_command)

    results_path = args.results or "results/tables/main_experiment_results.csv"
    summary_path = args.summary or "results/tables/summary_results.csv"

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
                args.analysis_dir,
            ),
            _python_command(
                "scripts/make_plots.py",
                "--input",
                results_path,
                "--output-dir",
                args.figure_dir,
            ),
            _python_command("scripts/make_metadata_tables.py"),
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
        default="results/tables/analysis",
        help="Directory for analysis CSV files.",
    )
    parser.add_argument(
        "--figure-dir",
        default="results/figures",
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--validation-report",
        default="results/tables/validation_report.csv",
        help="Output CSV for validation report.",
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
