# LaTeX Report

`final_report.pdf` is the compiled manuscript; `main.tex`, `chapters/`, and
`references.bib` are its editable source. Citations use IEEEtran's numbered
BibTeX style. The single-column 11-point report layout is not an IEEE journal
submission template. The title-page identity and supervisor follow the proposal.

## Build

Run from the repository root; no local TeX installation is needed:

```bash
docker build -f report/Dockerfile -t csi6900-report .
docker run --rm --network none -v "$PWD:/work" csi6900-report
```

The Docker build downloads TeX packages once. Compilation is offline and creates
`report/build/main.pdf`; auxiliary files stay in that ignored directory.
GitHub Actions also compiles the report and rejects unresolved references and
layout warnings. That check uses committed assets and does not fetch the large
LFS snapshot or rerun the benchmark.
The tracked `final_report.pdf` is the reviewed build snapshot and is not replaced
automatically by arbitrary local builds. After reviewing a new build, update it
with `cp report/build/main.pdf report/final_report.pdf`.

## Regenerate Evidence

The committed `generated/` directory allows a clean clone to compile immediately.
Its numbers and plots come only from the audited 50f8995 result matrix. To
regenerate, first retrieve and verify the Git LFS snapshot using
`releases/README.md`, then run in the project Python environment:

```bash
python scripts/make_report_assets.py --release-root releases/restored
```

The script checks the frozen manifest, input hashes, and all eight recomputed
analysis tables. It rejects other result matrices instead of quietly mixing
releases. `generated/evidence.json` lists the input hashes. Report additions
do not change the original training code or claim that the models were retrained.
The historical audit must run from its restored snapshot, whose Git revision
matches its manifest, not from a later report revision.

`source_audit.csv` records primary reference URLs and what was checked. Research
references are separate from the historical report-format examples in
`reference_sources.csv`; those examples are not experimental evidence.

## Before Submission

Read the full report and confirm the supervisor's required format, submission
date, and institutional policy on AI assistance. A preparation disclosure is
included in the appendix. Do not claim significance at 0.05, universal method
superiority, or that the combined score beat confidence. All three would
contradict the experiment. The author remains responsible for understanding
and defending the manuscript; compilation and tests do not constitute approval.
