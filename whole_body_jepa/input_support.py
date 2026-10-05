"""Bind a fitted readout to its observed source input support."""

import numpy as np
import pandas as pd


def learn_input_support(frame, variables, *, max_age_hours):
    if not np.isfinite(max_age_hours) or max_age_hours < 0 or frame.empty:
        raise ValueError("input support requires nonempty development data and a valid age limit")
    counts = {}
    for variable in variables:
        value, age = f"{variable}_t", f"{variable}_age_hr"
        if value not in frame or age not in frame:
            counts[variable] = 0
            continue
        measured = pd.to_numeric(frame[value], errors="coerce")
        ages = pd.to_numeric(frame[age], errors="coerce")
        counts[variable] = int((np.isfinite(measured) & ages.between(0, max_age_hours)).sum())
    return {"variables": list(variables), "supported": [v for v in variables if counts[v] > 0],
            "unsupported": [v for v in variables if counts[v] == 0],
            "source_observation_counts": counts, "max_age_hours": float(max_age_hours)}


def apply_input_support(frame, contract):
    """Mask unsupported inputs on ALL rows before patient-history construction.

    Keep targets, timestamps, identity, and future labels unchanged. This is a
    model-specific input contract, not permission to discard clinical evidence.
    """
    variables = contract["variables"]
    allowed, denied = set(contract["supported"]), set(contract["unsupported"])
    if allowed & denied or allowed | denied != set(variables):
        raise ValueError("input support partition is invalid")
    result = frame.copy()
    for variable in contract["unsupported"]:
        result[f"{variable}_t"] = np.nan
        result[f"{variable}_age_hr"] = np.nan
    return result
