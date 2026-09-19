"""Build and certify the complete canonical code/results release."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SLUG = "dry_bean_random_forest_symmetric_0p2_confidence"


def _python_command(*args):
    return [sys.executable, *args]


def build_finalization_commands(args):
    """Return the ordered commands for one complete release build."""

    commands = []
    if not args.skip_download:
        commands.append(
            _python_command(
                "scripts/download_external_datasets.py",
                "--datasets",
                "dry_bean",
            )
        )

    commands.append(
        _python_command(
            "scripts/export_sklearn_datasets.py",
            "--include-external",
        )
    )

    pipeline = _python_command(
        "scripts/run_full_pipeline.py",
        "--config",
        args.config,
        "--skip-download",
        "--save-final-models",
        "--make-model-diagnostics",
        "--diagnostic-table-dir",
        f"results/tables/model_diagnostics/{ARTIFACT_SLUG}",
        "--diagnostic-figure-dir",
        f"results/figures/model_diagnostics/{ARTIFACT_SLUG}",
    )
    if args.skip_training:
        pipeline.append("--skip-training")
    if args.resume:
        pipeline.append("--resume")
    commands.append(pipeline)

    commands.append(
        _python_command(
            "scripts/audit_release.py",
            "--config",
            args.config,
            "--freeze",
        )
    )
    return commands


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse the existing external-dataset cache.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse the existing full result matrix.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted full matrix instead of starting over.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the finalization commands without running them.",
    )
    args = parser.parse_args()
    if args.skip_training and args.resume:
        parser.error("--skip-training and --resume cannot be used together")
    return args


def main():
    args = parse_args()
    for command in build_finalization_commands(args):
        print(f"Running: {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
