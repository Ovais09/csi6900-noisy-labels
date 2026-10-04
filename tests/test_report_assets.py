"""Regression checks for the report's pairing, provenance, and source contracts."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.make_report_assets import (
    CONDITIONS,
    RESULTS_SHA,
    build_report_assets,
    latex_table,
    paired_gains,
)

ROOT = Path(__file__).resolve().parents[1]


def test_latex_table_has_booktabs_and_explicit_row_terminators():
    table = latex_table(["Method", "Gain"], [["Confidence", "+4.27"]])
    assert table.startswith(r"\begin{tabular}{lr}")
    assert "Method & Gain \\\\\n" in table
    assert "Confidence & +4.27 \\\\\n" in table
    assert table.endswith("\\bottomrule\n\\end{tabular}\n")


def _paired_example():
    rows = []
    for seed, noisy, cleaned in [(42, .7, .8), (123, .9, .85)]:
        for condition, value in [("noisy_baseline", noisy), ("remove_suspicious", cleaned)]:
            rows.append({"dataset": "wine", "model": "random_forest", "noise_type": "symmetric",
                         "noise_rate": .2, "random_state": seed, "condition": condition,
                         "detector": "none" if condition == "noisy_baseline" else "confidence",
                         "accuracy": value})
    return pd.DataFrame(rows)


def test_report_gains_pair_exact_seeds_and_convert_to_percentage_points():
    result = paired_gains(_paired_example().sample(frac=1, random_state=9))
    result = result.sort_values("random_state")
    np.testing.assert_allclose(result.gain_pp, [10, -5])
    assert len(result) == 2


def test_duplicate_noisy_reference_is_not_silently_averaged():
    frame = _paired_example()
    with pytest.raises(pd.errors.MergeError):
        paired_gains(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


def test_report_rejects_a_different_release_before_writing(tmp_path):
    manifest = tmp_path / "release/results/logs/release_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    output = tmp_path / "generated"
    with pytest.raises(ValueError, match="immutable 50f8995"):
        build_report_assets(tmp_path / "release", output)
    assert not output.exists()


def test_committed_report_evidence_matches_snapshot_and_table_contract():
    generated = ROOT / "report/generated"
    evidence = json.loads((generated / "evidence.json").read_text())
    assert evidence["results_sha256"] == RESULTS_SHA
    assert evidence["result_rows"] == 3840
    assert len({record["path"] for record in evidence["inputs"]}) == len(evidence["inputs"])
    table = pd.read_csv(generated / "overall.csv")
    assert set(zip(table.condition, table.detector)) == set(CONDITIONS)
    assert table[["accuracy", "f1_macro", "auroc"]].notna().all().all()
    assert table[["accuracy", "f1_macro", "auroc"]].map(lambda value: 0 <= value <= 1).all().all()


def test_every_manuscript_citation_has_a_bibliography_and_verification_record():
    source = "\n".join(path.read_text() for path in (ROOT / "report").rglob("*.tex")
                       if "build" not in path.parts)
    cited = {key.strip() for block in re.findall(r"\\cite\{([^}]+)\}", source) for key in block.split(",")}
    bibliography = (ROOT / "report/references.bib").read_text()
    keys = re.findall(r"@\w+\{([^,]+),", bibliography)
    assert len(keys) == len(set(keys))
    assert cited == set(keys)
    audit = pd.read_csv(ROOT / "report/source_audit.csv")
    assert set(audit.bib_key) == cited
    assert audit.primary_source.str.startswith("https://").all()


def test_every_static_figure_reference_exists():
    for source in (ROOT / "report/chapters").glob("*.tex"):
        for filename in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source.read_text()):
            path = ROOT / "report/generated" / filename
            assert path.read_bytes().startswith(b"%PDF-"), filename
