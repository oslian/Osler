"""Unit-safe, last-known treatment snapshots, never inferred active infusions."""
import re

import numpy as np
import pandas as pd

from osler_jepa.treatment_units import (
    ACTION_DOSE_DIMENSIONS, ACTION_RATE_DIMENSIONS, canonical_dose, canonical_rate,
)

ACTIONS = ("insulin", "dextrose", "fluids", "systemic_steroid", "vasopressor")


def precise_columns():
    return tuple(
        f"hist_{action}_{kind}_{dimension}"
        for action in ACTIONS
        for kind, dimensions in (("last_rate", ACTION_RATE_DIMENSIONS), ("dose", ACTION_DOSE_DIMENSIONS))
        for dimension in dimensions.get(action, ())
    )


def rate_value(record):
    value = pd.to_numeric(record.get("rate_like"), errors="coerce")
    if not np.isfinite(value) or value < 0:
        return np.nan, None
    unit = record.get("rate_unit")
    if str(record.get("source")) == "eicu_infusiondrug":
        if record.get("rate_field") != "drugrate":
            return np.nan, None  # A fallback volume rate cannot inherit drug units.
        # eICU drugrate units are explicitly embedded in some drug labels.
        units = re.findall(r"\(([^()]+)\)", str(record.get("original_label", "")))
        valid = [item for item in units if canonical_rate(1., item)[1] is not None]
        if len(valid) != 1:
            return np.nan, None
        unit = valid[0]
    factor, dimension = canonical_rate(1., unit)
    return float(value * factor), dimension


def rate_channels(record, action):
    """Keep documented flow without pretending it establishes drug delivery."""
    value, dimension = rate_value(record)
    known = bool(np.isfinite(value))
    return {
        "action_rate": value if known and dimension in ACTION_RATE_DIMENSIONS.get(action, ()) else np.nan,
        "action_rate_dimension": dimension if known and dimension in ACTION_RATE_DIMENSIONS.get(action, ()) else None,
        "solution_flow_ml_per_hour": value if known and dimension == "volume_ml_per_hour" else np.nan,
    }


def snapshot_summary(anchors, evidence, lookback_hours=6.):
    """Last documented rate and completed dose; NaN means undocumented.

    Uses supplied event timestamps. Without storetime, this is retrospective
    evidence, not a guarantee of bedside availability. Never uses a subsequent
    snapshot to infer when a rate ended.
    """
    out = pd.DataFrame(np.nan, index=anchors.index, columns=precise_columns())
    if evidence.empty:
        return out
    allowed = {"eicu_infusiondrug", "mimic_inputevents", "mimic_emar"}
    groups = {int(k): g.sort_values("starttime") for k, g in evidence.groupby("stay_id")}
    for index, anchor in anchors.iterrows():
        group = groups.get(int(anchor.stay_id))
        if group is None:
            continue
        now = pd.Timestamp(anchor.t)
        known = group[(group.starttime <= now) &
                      (group.starttime > now - pd.Timedelta(hours=lookback_hours)) &
                      group.source.isin(allowed)]
        # Separate drug labels: never add rates with incompatible dimensions.
        for (action, _), records in known.groupby(["action", "original_label"]):
            if action not in ACTIONS:
                continue
            last = records.iloc[-1]
            value, dimension = rate_value(last)
            column = f"hist_{action}_last_rate_{dimension}"
            if column in out and np.isfinite(value):
                previous = out.at[index, column]
                out.at[index, column] = (previous if pd.notna(previous) else 0.) + value
        for _, event in known.iterrows():
            # Inputevents total amount is not known until the infusion ends.
            if event.source == "mimic_inputevents" and (pd.isna(event.endtime) or event.endtime > now):
                continue
            if event.source == "eicu_infusiondrug":
                continue  # Bag contents do not establish delivered dose.
            value, dimension = canonical_dose(event.get("amount_like"), event.get("amount_unit"))
            column = f"hist_{event.action}_dose_{dimension}"
            if column in out and np.isfinite(value):
                previous = out.at[index, column]
                out.at[index, column] = (previous if pd.notna(previous) else 0.) + value
    return out
