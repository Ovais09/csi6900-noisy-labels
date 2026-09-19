"""Shared structural contract for experiment result rows."""

from collections.abc import Iterable

import pandas as pd


BASE_CONDITIONS = ("clean_upper_bound", "noisy_baseline")
CLEANING_CONDITIONS = ("remove_suspicious", "relabel_suspicious")
COMBO_COLUMNS = ("dataset", "noise_type", "noise_rate", "model", "random_state")
ROW_KEY_COLUMNS = COMBO_COLUMNS + ("condition", "detector")


def combo_key(dataset, noise_type, noise_rate, model, random_state):
    """Normalize the fields that identify one experiment combination."""

    return (
        str(dataset),
        str(noise_type),
        float(noise_rate),
        str(model),
        int(random_state),
    )


def expected_row_signatures(detectors: Iterable[str]):
    """Return the required ``(condition, detector)`` rows for one combination."""

    detector_names = tuple(str(detector) for detector in detectors)
    signatures = {(condition, "none") for condition in BASE_CONDITIONS}
    signatures.update(
        (condition, detector)
        for detector in detector_names
        for condition in CLEANING_CONDITIONS
    )
    return signatures


def complete_combo_keys(df: pd.DataFrame, detectors: Iterable[str]):
    """Return keys whose rows exactly satisfy the configured result contract."""

    required = set(ROW_KEY_COLUMNS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Result CSV is missing resume columns: {missing}")

    expected = expected_row_signatures(detectors)
    complete = set()

    for raw_key, group in df.groupby(list(COMBO_COLUMNS), sort=False, dropna=False):
        signatures = list(zip(group["condition"], group["detector"]))
        if len(signatures) != len(expected):
            continue
        if len(signatures) != len(set(signatures)):
            continue
        if set(signatures) != expected:
            continue
        complete.add(combo_key(*raw_key))

    return complete
