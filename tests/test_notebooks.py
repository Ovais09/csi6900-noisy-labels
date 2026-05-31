"""Notebook file sanity checks."""

import json
from pathlib import Path


def test_notebooks_are_valid_json_and_python_cells_compile():
    notebook_paths = sorted(Path("notebooks").glob("*.ipynb"))

    assert notebook_paths
    for path in notebook_paths:
        with path.open(encoding="utf-8") as file:
            notebook = json.load(file)

        assert notebook["nbformat"] == 4
        assert notebook["cells"]

        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue

            source = "".join(cell["source"])
            compile(source, f"{path}:cell-{index}", "exec")
