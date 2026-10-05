"""Add anchor-time paCO2/paO2 context omitted by the legacy examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLES = ("paco2", "pao2")
EICU_NAMES = {"paCO2": "paco2", "paO2": "pao2"}
BOUNDS = {"paco2": (5.0, 200.0), "pao2": (10.0, 700.0)}


def _read_eicu_events(data_root: Path, stay_ids: set[int]) -> pd.DataFrame:
    parts = []
    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    for chunk in pd.read_csv(
        data_root / "lab.csv.gz", usecols=usecols, chunksize=1_000_000, low_memory=False
    ):
        selected = chunk[
            chunk["patientunitstayid"].isin(stay_ids)
            & chunk["labname"].isin(EICU_NAMES)
        ].copy()
        if selected.empty:
            continue
        selected["stay_id"] = selected["patientunitstayid"].astype(str)
        selected["variable"] = selected["labname"].map(EICU_NAMES)
        selected["value"] = pd.to_numeric(selected["labresult"], errors="coerce")
        selected["event_time"] = pd.Timestamp("2100-01-01") + pd.to_timedelta(
            pd.to_numeric(selected["labresultoffset"], errors="coerce"), unit="m"
        )
        parts.append(selected[["stay_id", "event_time", "variable", "value"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _read_mimic_events(measurement_events: Path, stay_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(
        measurement_events,
        columns=["encounter_id", "event_time", "variable", "value"],
        filters=[("variable", "in", list(VARIABLES))],
    ).rename(columns={"encounter_id": "stay_id"})
    frame["stay_id"] = frame["stay_id"].astype(str)
    return frame.loc[frame["stay_id"].isin(stay_ids)].copy()


def add_anchor_context(frame: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["stay_id"] = frame["stay_id"].astype(str)
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], errors="coerce")
    events = events.copy()
    events["stay_id"] = events["stay_id"].astype(str)
    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce")
    for variable in VARIABLES:
        values = np.full(len(frame), np.nan, dtype=np.float64)
        ages = np.full(len(frame), np.nan, dtype=np.float64)
        variable_events = events.loc[events["variable"].eq(variable)].copy()
        lo, hi = BOUNDS[variable]
        variable_events = variable_events.loc[
            pd.to_numeric(variable_events["value"], errors="coerce").between(lo, hi)
        ]
        groups = variable_events.groupby("stay_id", sort=False)
        for stay_id, positions in frame.groupby("stay_id", sort=False).indices.items():
            if stay_id not in groups.groups:
                continue
            right = groups.get_group(stay_id).sort_values("event_time")
            event_times = right["event_time"].map(lambda value: pd.Timestamp(value).value).to_numpy()
            event_values = right["value"].to_numpy(dtype=np.float64)
            for position in np.asarray(positions, dtype=np.int64):
                anchor_ns = pd.Timestamp(frame.iloc[position]["anchor_time"]).value
                index = int(np.searchsorted(event_times, anchor_ns, side="right") - 1)
                if index < 0:
                    continue
                age = (anchor_ns - event_times[index]) / 3.6e12
                if age <= 6.0:
                    values[position] = event_values[index]
                    ages[position] = age
        frame[f"{variable}_t"] = values
        frame[f"{variable}_age_hr"] = ages
        frame[f"_observed_{variable}"] = np.isfinite(values).astype(np.float32)
        frame[f"future_{variable}"] = np.nan
        frame[f"future_event_time_{variable}"] = pd.NaT
        frame[f"future_delta_t_hr_{variable}"] = np.nan
        frame[f"future_label_quality_{variable}"] = "missing"
        frame[f"future_label_weight_{variable}"] = 0.0
        frame[f"next_measurement_time_hr_{variable}"] = np.nan
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=("eicu", "mimiciv"), required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--measurement-events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.examples)
    stay_ids = set(frame["stay_id"].astype(str))
    if args.database == "eicu":
        if args.data_root is None:
            raise ValueError("--data-root is required for eICU")
        events = _read_eicu_events(args.data_root, {int(value) for value in stay_ids})
    else:
        if args.measurement_events is None:
            raise ValueError("--measurement-events is required for MIMIC-IV")
        events = _read_mimic_events(args.measurement_events, stay_ids)
    output = add_anchor_context(frame, events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    report = {
        "schema": "blood_gas_anchor_context.v1",
        "database": args.database,
        "rows": int(len(output)),
        "patients": int(output["subject_id"].nunique()),
        "observed_rows": {
            variable: int(output[f"_observed_{variable}"].sum()) for variable in VARIABLES
        },
        "future_blood_gas_used": False,
        "output": str(args.output.resolve()),
        "causal_claim_allowed": False,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
