"""Reproducibility helpers."""

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy random generators."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
