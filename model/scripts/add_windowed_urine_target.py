"""Build a body-weight-normalized urine target over a real time window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "urine_output_3h_per_kg_hour"


def _collection_interval(value: object) -> float:
    try:
        payload = json.loads(value) if isinstance(value, str) else (value or {})
        interval = float(payload.get("collection_interval_hr", 1.0))
    except (TypeError, ValueError, json.JSONDecodeError):
        interval = 1.0
    return float(np.clip(interval, 0.05, 6.0))


def _window_rate(
    event_end: np.ndarray,
    event_start: np.ndarray,
    rates: np.ndarray,
    start: float,
    stop: float,
    minimum_coverage_fraction: float,
) -> tuple[float, float]:
    selected = (event_end > start) & (event_end <= stop) & (event_start < stop)
    if not selected.any():
        return np.nan, 0.0
    overlap = np.minimum(event_end[selected], stop) - np.maximum(event_start[selected], start)
    overlap = np.maximum(overlap, 0.0)
    coverage = float(np.sum(overlap))
    width = float(stop - start)
    if coverage < width * minimum_coverage_fraction:
        return np.nan, coverage
    return float(np.sum(rates[selected] * overlap) / coverage), coverage


def add_windowed_target(
    frame: pd.DataFrame,
    urine_events: pd.DataFrame,
    *,
    window_hours: float = 3.0,
    minimum_coverage_fraction: float = 2.0 / 3.0,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], errors="coerce")
    frame["stay_id"] = frame["stay_id"].astype(str)
    urine_events = urine_events.copy()
    urine_events["encounter_id"] = urine_events["encounter_id"].astype(str)
    urine_events["event_time"] = pd.to_datetime(urine_events["event_time"], errors="coerce")
    urine_events["value"] = pd.to_numeric(urine_events["value"], errors="coerce")
    urine_events["interval_hr"] = urine_events["metadata"].map(_collection_interval)
    urine_events = urine_events.dropna(subset=["event_time", "value"])

    current = np.full(len(frame), np.nan, dtype=np.float64)
    future = np.full(len(frame), np.nan, dtype=np.float64)
    current_coverage = np.zeros(len(frame), dtype=np.float64)
    future_coverage = np.zeros(len(frame), dtype=np.float64)
    groups = urine_events.groupby("encounter_id", sort=False)
    for stay_id, positions in frame.groupby("stay_id", sort=False).indices.items():
        if stay_id not in groups.groups:
            continue
        events = groups.get_group(stay_id).sort_values("event_time")
        # Parquet may preserve datetime64[us]. Timestamp.value always returns
        # nanoseconds, keeping event and anchor clocks on the same scale.
        end = events["event_time"].map(lambda value: pd.Timestamp(value).value).to_numpy(
            dtype=np.float64
        ) / 3.6e12
        interval = events["interval_hr"].to_numpy(dtype=np.float64)
        start = end - interval
        rates = events["value"].to_numpy(dtype=np.float64)
        for position in np.asarray(positions, dtype=np.int64):
            anchor = pd.Timestamp(frame.iloc[position]["anchor_time"]).value / 3.6e12
            current_rate, current_hours = _window_rate(
                end, start, rates, anchor - window_hours, anchor, minimum_coverage_fraction
            )
            future_rate, future_hours = _window_rate(
                end, start, rates, anchor, anchor + window_hours, minimum_coverage_fraction
            )
            current[position] = current_rate
            future[position] = future_rate
            current_coverage[position] = current_hours
            future_coverage[position] = future_hours

    weight = pd.to_numeric(frame["body_weight_t"], errors="coerce")
    valid_weight = weight.between(25.0, 300.0)
    frame[f"{TARGET}_t"] = current / weight
    frame[f"{TARGET}_age_hr"] = 0.0
    frame[f"_observed_{TARGET}"] = (np.isfinite(current) & valid_weight).astype(np.float32)
    frame[f"future_{TARGET}"] = future / weight
    frame[f"future_event_time_{TARGET}"] = frame["anchor_time"] + pd.to_timedelta(
        window_hours, unit="h"
    )
    frame[f"future_delta_t_hr_{TARGET}"] = float(window_hours)
    strict = np.isfinite(current) & np.isfinite(future) & valid_weight
    frame[f"future_label_quality_{TARGET}"] = np.where(strict, "strict", "missing")
    frame[f"future_label_weight_{TARGET}"] = strict.astype(np.float32)
    frame[f"next_measurement_time_hr_{TARGET}"] = np.nan
    frame[f"_coverage_hours_{TARGET}_t"] = current_coverage
    frame[f"_coverage_hours_future_{TARGET}"] = future_coverage
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--measurement-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-coverage-fraction", type=float, default=2.0 / 3.0)
    args = parser.parse_args()

    frame = pd.read_parquet(args.examples)
    events = pd.read_parquet(
        args.measurement_events,
        columns=["encounter_id", "event_time", "variable", "value", "metadata"],
        filters=[("variable", "=", "urine_output")],
    )
    output = add_windowed_target(
        frame,
        events,
        minimum_coverage_fraction=args.minimum_coverage_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    strict = output[f"future_label_weight_{TARGET}"].eq(1.0)
    report = {
        "schema": "windowed_weight_normalized_urine.v1",
        "rows": int(len(output)),
        "patients": int(output["subject_id"].nunique()),
        "strict_rows": int(strict.sum()),
        "strict_patients": int(output.loc[strict, "subject_id"].nunique()),
        "minimum_coverage_fraction": float(args.minimum_coverage_fraction),
        "window_hours": 3.0,
        "target": TARGET,
        "output": str(args.output.resolve()),
        "causal_claim_allowed": False,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
