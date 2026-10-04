# CSI6900 Noisy Labels

Reproducible experiments for lightweight noisy-label detection and cleaning on
tabular classification datasets.

The pipeline:

1. Load public tabular datasets, treating their supplied labels as the reference.
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

For development and code-quality checks, install the pinned audit tools too:

```bash
pip install -r requirements-dev.txt
```

## Docker

Start Docker Desktop first if you are on macOS.

Build the image:

```bash
docker build \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t csi6900-noisy-labels .
```

The revision build argument lets manifests created inside the image identify
the exact source commit even though `.git` is excluded from the build context.
The multi-architecture base image is pinned by digest. `requirements-docker.lock`
pins the resolved Python 3.11 container dependencies, including Jupyter's
dependencies. Local Python 3.9 installations use `requirements.txt` instead.
Native numerical threads are bounded to avoid oversubscription; random forests
can still use the CPUs assigned to the container.

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

Run the quick end-to-end pipeline in the container:

```bash
docker run --rm \
  -v "$PWD/datasets:/app/datasets" \
  -v "$PWD/results:/app/results" \
  csi6900-noisy-labels \
  python scripts/run_full_pipeline.py --quick --results results/tables/docker_pipeline_quick_results.csv --figure-dir results/figures/docker_pipeline_quick
```

Run the full training/results pipeline in the container:

```bash
docker run --rm \
  -v "$PWD/datasets:/app/datasets" \
  -v "$PWD/results:/app/results" \
  csi6900-noisy-labels \
  python scripts/run_full_pipeline.py
```

Build the complete final artifact set and freeze its checksums:

```bash
python scripts/finalize_release.py
```

To rebuild derived outputs, models, and diagnostics from an existing complete
matrix without retraining every combination:

```bash
python scripts/finalize_release.py --skip-download --skip-training
```

## Quick Verification

Run one small experiment combination:

```bash
python scripts/run_full_pipeline.py --quick --skip-download
```

Quick mode writes only to `quick_pipeline_*` paths, so it cannot overwrite the
full experiment CSV, summary tables, analysis tables, or figures.

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

Dry Bean is downloaded from the UCI Machine Learning Repository: 13,611 source
samples, 16 numeric features, and 7 classes. The benchmark removes 68 repeated
feature rows before any split, leaving 13,543 observations. This prevents
duplicate observations from crossing the holdout or cross-validation boundaries.
The original parsed data remains in `datasets/processed/dry_bean_source.csv`;
the exported benchmark is `datasets/processed/dry_bean.csv`.

Run tests:

```bash
pytest -q
```

Run static analysis and branch coverage:

```bash
ruff check src scripts tests main.py
coverage erase
coverage run --branch --source=src,scripts -m pytest -q
coverage report --include='src/*' --show-missing --fail-under=85
```

GitHub Actions runs the same lint and coverage checks on Python 3.9 and 3.11,
then builds the Docker image, runs its test suite, and executes the offline
quick pipeline. This workflow is configured here; a green remote run still
requires pushing the commits and checking GitHub Actions.

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
python scripts/run_full_pipeline.py
```

To regenerate derived files from an existing result CSV without retraining:

```bash
python scripts/run_full_pipeline.py --skip-download --skip-training
```

To print the planned commands without running them:

```bash
python scripts/run_full_pipeline.py --dry-run
```

To save fitted final models and create model-level diagnostics for one
representative setting:

```bash
python scripts/run_full_pipeline.py \
  --skip-download \
  --skip-training \
  --save-final-models \
  --make-model-diagnostics \
  --artifact-dataset dry_bean \
  --artifact-model random_forest \
  --artifact-noise-type symmetric \
  --artifact-noise-rate 0.2 \
  --artifact-detector confidence
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
- `results/tables/analysis/paired_cleaning_comparisons.csv`
- `results/tables/analysis/combined_detector_comparisons.csv`
- `results/tables/model_diagnostics/*.csv`
- `results/figures/*.png`
- `results/figures/*.pdf`
- `results/figures/model_diagnostics/*.png`
- `results/logs/run_manifest.json`
- `results/models/**/clean_upper_bound.joblib`
- `results/models/**/noisy_baseline.joblib`
- `results/models/**/remove_suspicious.joblib`
- `results/models/**/relabel_suspicious.joblib`

`scripts/validate_results.py` checks the saved CSV against
`configs/experiment_config.yaml`. It fails loudly if combinations are missing,
row counts are wrong, duplicate result rows exist, or metric columns contain
invalid values.

`scripts/make_analysis_tables.py` computes cleaning gains after pairing each
cleaned result with the noisy baseline from the same dataset, noise type, noise
rate, model, and seed. Seeds are averaged within settings, then settings within
datasets. Only dataset means are used for bootstrap intervals and exact
two-sided paired sign-flip tests, with equal weight for each dataset. Holm
adjustment covers the comparisons within each scope and metric. Per-dataset
rows are descriptive only: their inferential fields are blank. With just four
datasets, the smallest attainable two-sided exact p-value is 0.125; these
results cannot establish population-wide significance at 0.05. Bootstrap
intervals with four datasets are exploratory, not strong uncertainty guarantees.
Win/tie/loss rates and median gains summarize settings, not independent trials.
The separate combined-detector table pairs combined against each simpler
detector under the same removal or relabeling condition.

The dataset-level comparison principle follows
[Demsar (2006)](https://jmlr.org/papers/v7/demsar06a.html); the implemented exact
test is SciPy's [paired sign-flip permutation test](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats.permutation_test.html),
not Demsar's complete testing procedure.

`scripts/make_run_manifest.py` records SHA-256 fingerprints for the config,
datasets, and result CSV together with package versions, Python/platform
details, and the Git revision. Use this file to identify exactly what produced
a result set.

Training writes a `*.csv.training.json` sidecar at every completed combination.
Resume verifies the result checksum, training source files, dataset contents,
configuration (including CV folds), Python version, platform, and dependencies.
Changed or untracked checkpoints must be retrained. Regenerating derived outputs
preserves the original training provenance instead of attributing old results to
the current reporting environment.

`scripts/make_model_diagnostics.py` retrains one selected setting and saves
confusion matrices, ROC curves, precision-recall curves, macro-F1 summaries,
and per-class classification reports. `scripts/train_final_models.py` saves the
four fitted comparison models for that same setting.

`scripts/audit_release.py` independently recomputes summary and analysis
tables, validates every canonical PNG and its vector PDF companion, reloads
saved models and matches them to the matrix, recomputes classification reports
and curve areas, reconstructs splits and corruption counts, verifies exported
datasets against their source archive, checks provenance, and freezes a
SHA-256 inventory. Run `python scripts/audit_release.py` after freezing to
detect any changed or missing artifact.

Generated result files are intentionally ignored by git.

## Research Boundaries

- Validation is reserved, not used for fitting, selecting detectors, or tuning
  thresholds. All reported predictions use untouched test labels. Scaling is
  fitted inside each training/OOF pipeline, not before the split.
- Selecting the known injected noise fraction is an **oracle budget**. This
  does not demonstrate deployment with unknown real-world noise rates.
- `clean_upper_bound` is a legacy column value for a clean-label reference,
  not a guaranteed upper bound. Original dataset labels are not independently
  verified as error-free.
- The cyclic noise model is class-dependent but is not instance-dependent or
  a realistic confusion matrix learned from annotators. In binary tasks both
  noise mechanisms flip to the other class; only sampling and rounding differ.
- Combined is an empirical heuristic, not a demonstrated novel or universally
  superior algorithm. In binary classification its default score is monotone
  in confidence suspiciousness away from probability ties, so it generally
  produces the same ranking. Multiclass ablations are essential.
- Disagreement produces many tied scores. Selection uses deterministic row
  order to break ties. Margin measures uncertainty, not necessarily label error.
- Relabeling uses one model's out-of-fold predictions, not multi-model consensus.
- The best-by-setting table is retrospective test-set description, not an
  unbiased model-selection procedure. Saved-model diagnostics illustrate one
  setting; the complete matrix is the comparative evidence.
- Diagnostic multiclass ROC/PR curves use micro averaging. The main AUROC is
  macro one-vs-rest. PR AUC uses trapezoidal integration, not average precision.
- Four datasets and four overlapping random splits limit generalization. The
  benchmark is CPU-based; Docker isolates dependencies, not physical hardware.

## Preserve The Release

From a clean checkout, use the same Docker image to finalize and re-audit:

```bash
docker run --rm --network none \
  -v "$PWD/datasets:/app/datasets" -v "$PWD/results:/app/results" \
  csi6900-noisy-labels \
  python scripts/finalize_release.py --skip-download --skip-training

docker run --rm --network none \
  -v "$PWD/datasets:/app/datasets:ro" -v "$PWD/results:/app/results" \
  csi6900-noisy-labels python scripts/audit_release.py

.venv/bin/python scripts/package_release.py
.venv/bin/python scripts/package_release.py --verify results/logs/csi6900-final-release.tar.gz
```

The bundle contains tracked source plus every frozen dataset, result, figure,
and fitted model. It excludes `.git`, the virtual environment, and old scratch
runs. `BUNDLE_MANIFEST.json` records all file checksums and `source_commit`.
Keep this archive in a separate backup location; pushing Git does **not** back
up these ignored artifacts. Only load trusted `.joblib` files: deserialization
can execute code.

To restore a trusted, verified bundle into an empty directory:

```bash
tar -xzf /path/to/csi6900-final-release.tar.gz -C /path/to/empty-directory
cd /path/to/empty-directory
docker build --build-arg VCS_REF="$(python3 -c 'import json; print(json.load(open("BUNDLE_MANIFEST.json"))["source_commit"])')" \
  -t csi6900-restored .
docker run --rm --network none \
  -v "$PWD/datasets:/app/datasets:ro" -v "$PWD/results:/app/results" \
  csi6900-restored python scripts/audit_release.py
```

The report handoff is in `report/report_outline.md`; verified examples and
their provenance are catalogued in `report/reference_sources.csv`. Neither
an old example nor this audit guarantees a grade or replaces supervisor review.

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
- The full result contract and reproducibility manifest validate successfully.
- The final release audit and frozen artifact inventory validate successfully.
