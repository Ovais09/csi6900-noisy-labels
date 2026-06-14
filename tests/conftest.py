"""Pytest bootstrap for local source imports.

The ``pytest`` console script can start with ``.venv/bin`` as ``sys.path[0]``,
while ``python -m pytest`` starts from the repository root. Add the project root
explicitly so both commands can import ``src`` and ``scripts`` consistently.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
