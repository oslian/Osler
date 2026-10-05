#!/usr/bin/env python3
"""Relabel one pre-anchor waveform cohort at clinically short horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_waveform_event_cohort import parse_master_header
from whole_body_jepa.waveform_event import (
    SustainedHypoxemiaDefinition,
    incident_sustained_hypoxemia,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    frame = pd.read_parquet(args.cohort)
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"])
    labels = {
        minutes: np.full(len(frame), np.nan, dtype=float) for minutes in args.horizon_minutes
    }
    descriptors = {
        minutes: {name: np.full(len(frame), np.nan, dtype=float) for name in (
            "future_spo2_min",
            "future_spo2_mean",
            "future_spo2_std",
            "future_low_fraction",
            "future_observation_count",
        )}
        for minutes in args.horizon_minutes
    }
    for record_index, (record, rows) in enumerate(frame.groupby("record", sort=True), start=1):
        first = rows.iloc[0]
        directory = Path(str(first["segment_path"])).parent
        numeric_path = directory / f"{record}n.csv.gz"
        master = parse_master_header(directory / f"{record}.hea")
        if master is None or not numeric_path.exists():
            continue
        numeric = pd.read_csv(numeric_path, usecols=lambda value: value in {"time", "SpO2 [%]"})
        times = pd.to_numeric(numeric["time"], errors="coerce").to_numpy(dtype=float) / 1000.0
        spo2 = pd.to_numeric(numeric["SpO2 [%]"], errors="coerce").to_numpy(dtype=float)
        for row_index, anchor_time in zip(rows.index, rows["anchor_time"]):
            anchor_seconds = (pd.Timestamp(anchor_time) - master["base_datetime"]).total_seconds()
            for minutes in args.horizon_minutes:
                definition = SustainedHypoxemiaDefinition(
                    horizon_seconds=float(minutes) * 60.0
                )
                label, descriptor = incident_sustained_hypoxemia(
                    times, spo2, anchor_seconds, definition
                )
                if label is None:
                    continue
                labels[minutes][row_index] = float(label)
                for name, value in descriptor.items():
                    descriptors[minutes][name][row_index] = float(value)
        if record_index % 30 == 0:
            print(f"records: {record_index}", flush=True)

    summary = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for minutes in args.horizon_minutes:
        output = frame.copy()
        output["event"] = labels[minutes]
        for name, values in descriptors[minutes].items():
            output[name] = values
        output = output[output["event"].notna()].copy()
        output["event"] = output["event"].astype("int8")
        path = args.output_dir / f"mimiciv_multimodal_hypoxemia_{minutes}m.parquet"
        output.to_parquet(path, index=False)
        summary[f"{minutes}m"] = {
            "rows": int(len(output)),
            "patients": int(output["subject_id"].nunique()),
            "events": int(output["event"].sum()),
            "prevalence": float(output["event"].mean()) if len(output) else None,
            "path": str(path),
        }
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--horizon-minutes", type=int, nargs="+", default=(15, 30))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
