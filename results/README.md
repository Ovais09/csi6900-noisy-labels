# Generated Results

Everything below `results/tables`, `results/figures`, `results/logs`, and
`results/models` is generated and intentionally ignored by Git. The `.gitkeep`
files preserve the directory structure in a clean clone.

## Core Outputs

- `tables/main_experiment_results.csv`: one row per condition and detector for
  every dataset/noise/rate/model/seed combination.
- `tables/summary_results.csv`: means and standard deviations across seeds.
- `tables/validation_report.csv`: structural and numerical integrity checks for
  the complete experiment matrix.
- `logs/run_manifest.json`: config, dataset, result, environment, and Git
  fingerprints for reproducibility.
- `logs/release_manifest.json`: frozen sizes and SHA-256 hashes for every
  canonical dataset, table, figure, diagnostic, model, and provenance file.
- `logs/final_audit.json`: pass/fail certification for the complete code/results
  release.

## Analysis Tables

- `tables/analysis/condition_summary.csv`: classification performance by
  dataset, condition, and detector.
- `tables/analysis/detector_ranking.csv`: overall label-error detection ranking
  with each detector counted once per experimental run.
- `tables/analysis/detector_breakdown.csv`: detector precision, recall, and F1
  by dataset and noise type.
- `tables/analysis/cleaning_gain_summary.csv`: unadjusted run-level gains over
  the noisy baseline.
- `tables/analysis/noise_impact_summary.csv`: clean-to-noisy accuracy loss.
- `tables/analysis/best_cleaning_by_setting.csv`: best cleaning strategy for
  each dataset/noise/rate/model setting.
- `tables/analysis/paired_cleaning_comparisons.csv`: seed-averaged paired
  effects, bootstrap 95% intervals, win/tie/loss rates, paired effect sizes,
  and Holm-adjusted Wilcoxon p-values.

## Figures and Models

- `figures/*.png`: accuracy, macro-F1, and detection-F1 curves.
- `figures/model_diagnostics/`: confusion matrices, ROC curves,
  precision-recall curves, and condition-level F1 summaries.
- `models/`: optional fitted estimators plus model hashes and metadata.

Regenerate all derived outputs from an existing full result CSV with:

```bash
python scripts/run_full_pipeline.py --skip-download --skip-training
```

Regenerate and certify the complete final release with:

```bash
python scripts/finalize_release.py --skip-download --skip-training
```
