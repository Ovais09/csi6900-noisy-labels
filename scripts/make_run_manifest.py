"""Create a machine-readable provenance manifest for experiment results."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_experiment_config, quick_experiment_config
from src.utils.io import save_json_atomic
from src.utils.provenance import build_run_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument(
        "--results",
        default="results/tables/main_experiment_results.csv",
    )
    parser.add_argument(
        "--output",
        default="results/logs/run_manifest.json",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.quick:
        config = quick_experiment_config(config)
    manifest = build_run_manifest(
        config=config,
        config_path=args.config,
        results_path=args.results,
        project_root=PROJECT_ROOT,
    )
    path = save_json_atomic(manifest, args.output)
    print(f"Saved run manifest to {path}")


if __name__ == "__main__":
    main()
