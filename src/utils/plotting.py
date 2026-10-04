"""Consistent raster and vector exports for report figures."""

from pathlib import Path

CONDITION_LABELS = {
    "clean_upper_bound": "Clean reference",
    "noisy_baseline": "Noisy baseline",
    "remove_suspicious": "Remove",
    "relabel_suspicious": "Relabel",
}


def save_report_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
