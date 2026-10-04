# Frozen Code And Results

`code-results-50f8995.tar.gz` is the immutable, audited code/results release.
It is tracked through Git LFS, not as a large ordinary Git blob. It contains
264 files, including the source snapshot, all 3,840 result rows, datasets,
four saved classifiers, PNG/PDF figures, and provenance/checksum inventories.
Exploratory outputs and older backup archives are deliberately excluded.

- Source revision: `50f8995a014f4d68714cba7a4ea81e911c859374`
- Original training revision: `2d0dfbf34480bba5e602f165fd12dc0c3ef75ef8`
- Archive SHA-256: `73b15ce8a707f671f7d7102d8ac776d504a9786afda5d933f0a72e7363861e58`
- Result CSV SHA-256: `2576804e3caadd3025ff95a9835729ced219fd38dce4ca30ee4a399727a61150`

## Retrieve And Verify

Install Git LFS (`brew install git-lfs` on macOS), then run from the repo root:

```bash
git lfs install --local
git lfs pull
python scripts/package_release.py --verify releases/code-results-50f8995.tar.gz
mkdir -p releases/restored
tar -xzf releases/code-results-50f8995.tar.gz -C releases/restored
docker build --build-arg VCS_REF=50f8995a014f4d68714cba7a4ea81e911c859374 \
  -t csi6900-release releases/restored
docker run --rm --network none \
  -v "$PWD/releases/restored/datasets:/app/datasets:ro" \
  -v "$PWD/releases/restored/results:/app/results:ro" \
  csi6900-release python scripts/audit_release.py --output /tmp/audit.json
```

Only extract into an empty directory. Do not unpack over your working tree.
The historical release audit checks its own revision, not later report commits.
The report's regeneration script accepts this restored directory as `--release-root`.

`git push` invokes the local LFS hook to upload the object as well as the Git
pointer. A local commit alone is not a remote backup. LFS storage and bandwidth
are subject to the repository owner's GitHub quota; no paid storage is enabled
by this project. GitHub ZIP downloads may contain only a pointer, so use a
Git-LFS-enabled clone or `git lfs pull`.

## Data Attribution

Original datasets retain their source terms; this archive does not relicense
third-party data. Dry Bean (M. Koklu and I. A. Ozkan) is distributed by UCI under
CC BY 4.0: https://doi.org/10.24432/C50S4B. Its original archive is preserved;
the benchmark removes 68 duplicate feature rows. Breast Cancer Wisconsin
(Diagnostic) and Wine are distributed through scikit-learn from UCI, and
Digits is scikit-learn's 1,797-row optical-digit subset. Source citations and
the transformations used appear in `report/references.bib` and the report.
Never load an untrusted joblib file: model deserialization can execute code.
