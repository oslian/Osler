"""Exact next-measurement-time labels from timestamped event tables.

Fixed-horizon observation probability and time-to-next-measurement are
different targets. This module only creates the latter when raw event
timestamps are available; transition grids without raw events are marked
unavailable instead of being treated as exact measurement labels.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def next_measurement_labels(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    anchor_time_col: str = "t",
    event_time_col: str = "charttime",
    value_col: str | None = None,
) -> pd.DataFrame:
    """Attach exact hours to the first event strictly after each anchor.

    ``events`` must contain real measurement rows, not a regularized transition
    grid. ``value_col`` can be used to exclude rows whose target value is
    missing. Rows with no future event retain NaN and ``observed=False``.
    """
    required = {subject_col, anchor_time_col}
    if not required.issubset(anchors.columns):
        raise KeyError(f"anchors missing {sorted(required - set(anchors.columns))}")
    required_events = {subject_col, event_time_col}
    if not required_events.issubset(events.columns):
        raise KeyError(f"events missing {sorted(required_events - set(events.columns))}")
    left = anchors[[subject_col, anchor_time_col]].copy()
    left[anchor_time_col] = pd.to_datetime(left[anchor_time_col], errors="coerce")
    right = events[[subject_col, event_time_col] + ([value_col] if value_col else [])].copy()
    right[event_time_col] = pd.to_datetime(right[event_time_col], errors="coerce")
    if value_col:
        right = right[pd.to_numeric(right[value_col], errors="coerce").notna()]
    left["_row_id"] = np.arange(len(left), dtype=np.int64)
    left = left.dropna(subset=[anchor_time_col]).copy()
    right = right.dropna(subset=[event_time_col]).copy()
    # Use integer nanoseconds rather than pandas datetime dtype directly.
    # Different pandas/Arrow readers may materialize one side as ns and the
    # other as us; merge_asof rejects that despite identical timestamps.
    left["_anchor_ns"] = left[anchor_time_col].astype("datetime64[ns]").astype("int64") + 1
    right["_event_ns"] = right[event_time_col].astype("datetime64[ns]").astype("int64")
    left = left.sort_values([subject_col, "_anchor_ns"])
    right = right.sort_values([subject_col, "_event_ns"])
    merged = pd.merge_asof(
        left, right, by=subject_col, left_on="_anchor_ns",
        right_on="_event_ns", direction="forward", allow_exact_matches=True,
    ).sort_values("_row_id")
    delta = (merged["_event_ns"] - merged["_anchor_ns"]) / (3.6e12)
    return pd.DataFrame({
        "next_measurement_time_hr": delta.to_numpy(dtype=np.float64),
        "next_measurement_observed": np.isfinite(delta.to_numpy(dtype=np.float64)),
        "next_measurement_label_source": "raw_timestamped_event",
    })


def unavailable_next_measurement_labels(rows: int, reason: str) -> pd.DataFrame:
    """Explicit fail-closed output for transition files without raw events."""
    return pd.DataFrame({
        "next_measurement_time_hr": np.full(int(rows), np.nan),
        "next_measurement_observed": np.zeros(int(rows), dtype=bool),
        "next_measurement_label_source": f"unavailable:{reason}",
    })
