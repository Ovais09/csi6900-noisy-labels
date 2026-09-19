# Final Report Outline

Use this outline to turn the proposal and generated experiment artifacts into
the CSI6900 final report.

## 1. Introduction

Goal: explain why noisy labels matter for tabular classification.

Include:

- Supervised tabular classifiers assume training labels are correct.
- Real labels can contain human error, ambiguous definitions, or collection
  inconsistencies.
- The project studies robustness under controlled symmetric and class-dependent
  label noise.

Evidence to cite:

- Nazaretyan et al. 2025 benchmark: no single mislabeled-data method is best
  across settings.
- Yang et al. 2024 EHR paper: label noise is underexplored for tabular health
  data.
- Li and Zhu 2024 survey: noisy-label processing is broad, but image methods do
  not transfer automatically to tabular settings.

## 2. Related Work

Organize by theme:

- General noisy-label processing surveys.
- Tabular noisy-label benchmarks.
- Gradient boosted decision trees under label noise.
- EHR/tabular domain label-noise studies.
- Evaluation challenges with synthetic or already-noisy datasets.

End this section with the gap:

Lightweight tabular methods are practical, but their performance varies by
dataset, model, noise type, and noise rate. A controlled benchmark is needed.

## 3. Proposed Approach

Describe the pipeline:

- Load tabular datasets.
- Split into train, validation, and test.
- Inject label noise only into training labels.
- Train clean and noisy baselines.
- Score suspicious labels using confidence, disagreement, margin, and combined
  detectors.
- Select the top `noise_rate` fraction as suspicious.
- Retrain after either removing or relabeling suspicious samples.

Original contribution:

The combined suspicious-label score:

```text
combined = 0.5 * confidence + 0.3 * disagreement + 0.2 * margin
```

Generated tables to use:

- `results/tables/dataset_overview.csv`
- `results/tables/model_overview.csv`
- `results/tables/noise_settings.csv`
- `results/tables/detector_methods.csv`
- `results/tables/cleaning_conditions.csv`

## 4. Experimental Setup

Report:

- Datasets: Breast Cancer, Wine, Digits, and UCI Dry Bean.
- Models: logistic regression, random forest, histogram gradient boosting.
- Noise types: symmetric and class-dependent.
- Noise rates: 10%, 20%, 30%, 40%.
- Seeds: 42, 123, 999, 2026.
- Metrics: accuracy, macro F1, AUROC, detection precision, detection recall,
  detection F1.

Generated results:

- `results/tables/main_experiment_results.csv`
- `results/tables/summary_results.csv`
- `results/logs/run_manifest.json`

## 5. Results and Analysis

Answer these questions:

- How much does noisy training hurt compared with clean training?
- Does removal improve over the noisy baseline?
- Does relabeling improve over the noisy baseline?
- Which detector has the best corrupted-label detection F1?
- Does the combined detector outperform or match simpler detector baselines?
- Where does cleaning fail or reduce performance?

Useful figures:

- Accuracy vs noise rate.
- Macro F1 vs noise rate.
- Detection F1 vs noise rate.

Statistical evidence:

- Use `results/tables/analysis/paired_cleaning_comparisons.csv` for paired
  cleaning-vs-noisy gains, bootstrap 95% confidence intervals, win rates, and
  Holm-adjusted Wilcoxon p-values.
- Explain that the four seeds are averaged within each dataset/noise/rate/model
  setting before inferential comparisons, rather than treated as independent
  settings.

## 6. Conclusion

Summarize:

- Main empirical findings.
- Whether lightweight cleaning improved robustness.
- Which models/noise types were easiest or hardest.
- Limitations: mostly synthetic corruption, known-noise-rate selection,
  cyclic class-dependent transitions, and no hyperparameter tuning.
- Future work: larger OpenML datasets, real noisy labels, stronger tabular
  models, adaptive threshold selection.

## Rubric Mapping

- Presentation quality: use generated tables and figures, keep sections clear.
- Introduction: Section 1.
- Related work: Section 2.
- Proposed approach: Section 3.
- Evaluation and analysis: Sections 4 and 5.
- Conclusion and future work: Section 6.
