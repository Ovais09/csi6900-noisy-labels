# Final Report Outline

Use this outline to turn the proposal and generated experiment artifacts into
the CSI6900 final report.

## Verified Examples And Formatting

The approved proposal and Thomas Tran's marking scheme govern this project.
The university asks for a concrete research deliverable and says literature
summaries alone are insufficient. Supervisor-specific requirements can take
precedence over general formatting suggestions.
[Official project guidance](https://www.uottawa.ca/faculty-engineering/student-hub/report-project-masters-program).

The general guidelines suggest 10-12 point type, letter portrait pages,
one-inch margins, consistent numbered tables/figures, and an abstract of at
most 150 words without references or figures. Prefer symbols and line styles
that survive grayscale printing. Include title, abstract, contents, table/figure
lists as appropriate, main chapters, appendices, and references.
[Submission guidelines](https://www.uottawa.ca/faculty-engineering/student-hub/report-project-masters-program/submission-guidelines).

Two full-text examples informed this outline:

- Yuan Gao, *InstanceNet: object instance segmentation using DNN* (2020),
  explicitly headed CSI6900 Project. Its method, training details, component
  comparisons, sensitivity table, and limitations are useful structural
  examples. Its supervisor and assessed status are not established; it is an
  author-posted report, not evidence of Tran's marking preferences.
  [Full text](https://vixra.org/pdf/2004.0318v1.pdf).
- Raza Ul Haq, *Hybrid Recommender System towards User Satisfaction* (2013).
  Its acknowledgments identify Thomas Tran as thesis supervisor. It separates
  related work, proposed model, implementation, evaluation, and conclusions.
  This is a **thesis**, not a verified CSI6900 report; do not copy its length
  or treat it as this course's required format.
  [Full text](https://www.collectionscanada.gc.ca/obj/thesescanada/vol2/OOU/TC-OOU-24218.pdf).

Other historical CSI6900 reports were identified but not retrieved in full;
see `reference_sources.csv`. No full-text CSI6900 report specifically verified
as supervised by Tran was located. Do not imply otherwise. His supplied
supervision page was last updated in 2016 and is not a current course rubric.

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

Describe a reproducible benchmark and a simple combined-score hypothesis;
do not claim established algorithmic novelty or superiority merely from having
implemented a weighted sum. State the contribution supported by the evidence.

The combined suspicious-label score:

```text
combined = 0.5 * confidence + 0.3 * disagreement + 0.2 * margin
```

Define every term using training-only out-of-fold probabilities. Document
argmax label replacement, exact top-k rounding, deterministic tie-breaking,
and the known-noise-rate selection budget. Relabeling is not an ensemble
consensus. Explain why binary combined/confidence rankings generally coincide
and why multiclass comparisons are the informative ablation.

Include a short implementation/reproducibility subsection: tested Python
modules, thin notebook entry points, pinned Docker runtime, atomic checkpoints,
dataset fingerprints, original training commit, and frozen release checksums.

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
- Dry Bean: 13,611 source rows; 68 exact duplicates removed before splitting;
  13,543 benchmark rows. Retain the original source and cite the
  [UCI dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).
- Stratified 60/20/20 split; validation reserved but unused. No preprocessing
  sees held-out rows. The original labels are a reference, not certified truth.
- Cyclic class-dependent corruption is a simplified transition model. In
  binary classification it has the same label-flip destination as symmetric
  noise; its sample selection is per-class rather than global.
- 384 experiment combinations and 3,840 rows; repeated clean baselines and
  removal/relabel detection values are not independent observations.

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
  cleaning-vs-noisy gains and `combined_detector_comparisons.csv` for matched
  comparisons to each simpler scorer.
- Average seeds within settings, then settings within datasets. Overall
  bootstrap intervals and exact sign-flip tests use four dataset-level gains,
  not 96 correlated settings or 384 overlapping random splits. Holm adjustment
  is within metric/scope. Per-dataset intervals and p-values are undefined.
- With four datasets the minimum exact two-sided p-value is 0.125. Treat the
  study and its bootstrap intervals as exploratory; do not claim statistical
  significance at 0.05 or population-wide superiority. Explain any disagreement
  between exploratory bootstrap intervals and the discrete exact tests.
- Win rates and median gains describe settings; they are not hypothesis tests.
  Report negative results and ties, especially combined versus confidence.
- Main AUROC is macro one-vs-rest; diagnostic multiclass curves are micro
  averaged. PR AUC is trapezoidal, not average precision. The diagnostic setting
  is illustrative, not independent evidence that this is the best model.

Prefer vector `.pdf` plots for typesetting. Show selected main-text figures and
move the complete per-setting tables to the appendix. Never use the
retrospective best-by-setting table as evidence of an unbiased selection rule.

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
