# CSI6900 Noisy Labels

Reproducible experiments for lightweight noisy-label detection and cleaning on
tabular classification datasets.

The pipeline:

1. Load clean tabular datasets.
2. Split into train/validation/test.
3. Corrupt only the training labels.
4. Train clean and noisy baselines.
5. Detect suspicious labels with confidence, disagreement, margin, and combined
   detector scores. The combined score is
   `0.5 * confidence + 0.3 * disagreement + 0.2 * margin`.
6. Compare removal and relabeling against the noisy baseline.
7. Save metrics, summary tables, validation reports, analysis tables, and figures.

## Project Structure

```text
configs/                 Experiment matrix configuration
datasets/                Optional raw/processed external data
results/tables/          Generated CSV outputs
results/figures/         Generated plots
scripts/                 Runnable experiment scripts
src/                     Reusable project code
notebooks/               Interactive Jupyter workflows
tests/                   Smoke and regression tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Docker

Start Docker Desktop first if you are on macOS.

Build the image:

```bash
docker build -t csi6900-noisy-labels .
```

If you see `failed to connect to the docker API at unix:///var/run/docker.sock`,
the Docker CLI is installed but no Docker backend is running. Fix it with one of:

```bash
# Docker Desktop option
brew install --cask docker
open -a Docker
docker info
```

```bash
# Lightweight Colima option
brew install colima
colima start --cpu 4 --memory 8 --disk 40
docker info
```

On Apple Silicon, if Docker Desktop shows `Rosetta installation failed`, click
`Disable Rosetta` in the Docker Desktop prompt. This project uses native ARM64
images, so Rosetta is not required.

Run tests in the container:

```bash
docker run --rm csi6900-noisy-labels
```

Run a quick training check in the container and save outputs back to your local
repo:

```bash
docker run --rm \
  -v "$PWD/datasets:/app/datasets" \
  -v "$PWD/results:/app/results" \
  csi6900-noisy-labels \
  python scripts/run_cleaning_experiments.py --quick --output results/tables/docker_quick_results.csv
```

Run the full training/results pipeline in the container:

```bash
docker run --rm \
  -v "$PWD/datasets:/app/datasets" \
  -v "$PWD/results:/app/results" \
  csi6900-noisy-labels \
  bash -lc "python scripts/download_external_datasets.py && python scripts/run_cleaning_experiments.py && python scripts/make_summary_tables.py && python scripts/make_analysis_tables.py && python scripts/make_plots.py && python scripts/make_metadata_tables.py && python scripts/validate_results.py"
```

## Quick Verification

Run one small experiment combination:

```bash
python scripts/run_cleaning_experiments.py --quick --output results/tables/quick_experiment_results.csv
python scripts/make_summary_tables.py --input results/tables/quick_experiment_results.csv --output results/tables/quick_summary_results.csv
python scripts/make_plots.py --input results/tables/quick_experiment_results.csv --output-dir results/figures/quick
```

Export the built-in datasets as local processed CSV files:

```bash
python scripts/export_sklearn_datasets.py
```

Download and prepare the external UCI Dry Bean dataset:

```bash
python scripts/download_external_datasets.py
```

Export all supported datasets, including Dry Bean:

```bash
python scripts/export_sklearn_datasets.py --include-external
```

Dry Bean is downloaded from the UCI Machine Learning Repository and adds a
larger real tabular benchmark with 13,611 samples, 16 numeric features, and 7
classes.

Run tests:

```bash
pytest -q
```

## Notebook Workflow

The notebooks are interactive entry points on top of the tested Python package.
They do not replace the `src/` code; they import it so command-line runs,
Docker runs, and notebook runs all use the same implementation.

- `notebooks/01_data_check.ipynb`: verify dataset loading, class balance, and
  train/validation/test splits.
- `notebooks/02_noise_and_detectors.ipynb`: inspect one noisy-label setting,
  detector scores, suspicious rows, and one full model result.
- `notebooks/03_run_experiments.ipynb`: run a small interactive experiment
  matrix, or flip `RUN_FULL_MATRIX = True` to use the YAML config.
- `notebooks/04_results_analysis.ipynb`: inspect saved results, rank detectors,
  measure cleaning gains, and save report-friendly analysis tables.

Start Jupyter from the project root:

```bash
jupyter notebook
```

## Full Experiment

The default matrix is defined in `configs/experiment_config.yaml`:

- Datasets: Breast Cancer, Wine, Digits, Dry Bean
- Models: logistic regression, random forest, histogram gradient boosting
- Noise types: symmetric, class-dependent
- Noise rates: 10%, 20%, 30%, 40%
- Detectors: combined, confidence, disagreement, margin
- Seeds: 42, 123, 999, 2026

Run:

```bash
python scripts/run_cleaning_experiments.py
python scripts/make_summary_tables.py
python scripts/make_analysis_tables.py
python scripts/make_plots.py
python scripts/make_metadata_tables.py
python scripts/validate_results.py
```

Main outputs:

- `results/tables/main_experiment_results.csv`
- `results/tables/summary_results.csv`
- `results/tables/dataset_overview.csv`
- `results/tables/model_overview.csv`
- `results/tables/noise_settings.csv`
- `results/tables/detector_methods.csv`
- `results/tables/cleaning_conditions.csv`
- `results/tables/validation_report.csv`
- `results/tables/analysis/*.csv`
- `results/figures/*.png`

`scripts/validate_results.py` checks the saved CSV against
`configs/experiment_config.yaml`. It fails loudly if combinations are missing,
row counts are wrong, duplicate result rows exist, or metric columns contain
invalid values.

Generated result files are intentionally ignored by git.

## Done Criteria

- Three datasets run successfully.
- At least one larger external tabular dataset runs successfully.
- Three baseline models run successfully.
- Symmetric and class-dependent noise are tested.
- Four noise rates are tested.
- Multiple random seeds are averaged.
- Clean, noisy, remove, and relabel conditions are saved.
- Classification metrics and detection metrics are saved.
- Summary CSV tables and plots are generated for the report.
