"""Build target-specific JEPA examples from a canonical event dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from osler_jepa.event_adapters import event_frame_to_ledger, read_event_dataset
from osler_jepa.event_dataset import build_event_examples, ledger_readiness_report
from osler_jepa.event_ledger import EventQuality, canonical_variable


def build_regular_anchors(
    events: pd.DataFrame,
    *,
    lookback_hours: float,
    anchor_step_hours: float,
    maximum_episode_hours: float,
) -> pd.DataFrame:
    measurements = events[events["kind"].eq("measurement")].copy()
    rows = []
    for encounter_id, group in measurements.groupby("encounter_id", sort=False):
        start = pd.Timestamp(group["event_time"].min())
        stop = min(
            pd.Timestamp(group["event_time"].max()),
            start + pd.Timedelta(hours=float(maximum_episode_hours)),
        )
        anchor = start + pd.Timedelta(hours=float(lookback_hours))
        first = group.iloc[0]
        while anchor <= stop:
            rows.append(
                {
                    "encounter_id": str(encounter_id),
                    "anchor_time": anchor,
                    "patient_id": str(first["patient_id"]),
                    "hospital_id": first.get("hospital_id"),
                    "care_unit": first.get("care_unit"),
                    "database": first.get("database"),
                }
            )
            anchor += pd.Timedelta(hours=float(anchor_step_hours))
    return pd.DataFrame.from_records(rows)


def build_measurement_anchors(
    events: pd.DataFrame,
    anchor_variable: str,
    *,
    lookback_hours: float,
    maximum_episode_hours: float,
) -> pd.DataFrame:
    """Use real target measurements as anchors without reading future events."""

    variable = canonical_variable(anchor_variable)
    measurements = events.loc[
        events["kind"].eq("measurement")
        & events["quality"].ne(EventQuality.INVALID.value)
    ].copy()
    if measurements.empty:
        return pd.DataFrame()
    start = measurements.groupby("encounter_id")["event_time"].min()
    candidates = measurements.loc[measurements["variable"].eq(variable)].copy()
    candidates["episode_start"] = candidates["encounter_id"].map(start)
    elapsed = (
        pd.to_datetime(candidates["event_time"])
        - pd.to_datetime(candidates["episode_start"])
    ).dt.total_seconds() / 3600.0
    candidates = candidates.loc[
        elapsed.between(float(lookback_hours), float(maximum_episode_hours))
    ].sort_values(["encounter_id", "event_time", "source_id"], kind="stable")
    candidates = candidates.drop_duplicates(["encounter_id", "event_time"])
    return pd.DataFrame(
        {
            "encounter_id": candidates["encounter_id"].astype(str),
            "anchor_time": pd.to_datetime(candidates["event_time"]),
            "hours_since_onset": elapsed.loc[candidates.index].astype(float),
            "patient_id": candidates["patient_id"].astype(str),
            "hospital_id": candidates["hospital_id"],
            "care_unit": candidates["care_unit"],
            "database": candidates["database"],
        }
    ).reset_index(drop=True)


def build_partitioned_examples(
    events: pd.DataFrame,
    anchors: pd.DataFrame,
    variables: tuple[str, ...],
    horizons: tuple[float, ...],
) -> pd.DataFrame:
    """Build examples one encounter at a time to bound object memory."""

    anchor_groups = {
        str(encounter_id): group
        for encounter_id, group in anchors.groupby("encounter_id", sort=False)
    }
    frames = []
    for encounter_id, group in events.groupby("encounter_id", sort=False):
        encounter_anchors = anchor_groups.get(str(encounter_id))
        if encounter_anchors is None or encounter_anchors.empty:
            continue
        ledger = event_frame_to_ledger(group)
        frames.append(
            build_event_examples(
                ledger,
                encounter_anchors,
                variables,
                horizons,
            )
        )
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else pd.DataFrame()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variables", default=None)
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--lookback-hours", type=float, default=6.0)
    parser.add_argument("--anchor-step-hours", type=float, default=3.0)
    parser.add_argument("--maximum-episode-hours", type=float, default=96.0)
    parser.add_argument(
        "--anchor-variable",
        default=None,
        help="Use real measurements of this variable as anchors instead of a regular grid.",
    )
    args = parser.parse_args()

    events = read_event_dataset(args.event_data)
    measurements = events[events["kind"].eq("measurement")]
    variables = (
        tuple(
            value.strip()
            for value in args.variables.split(",")
            if value.strip()
        )
        if args.variables
        else tuple(sorted(measurements["variable"].dropna().astype(str).unique()))
    )
    horizons = tuple(
        float(value.strip())
        for value in args.horizons.split(",")
        if value.strip()
    )
    if args.anchor_variable:
        anchors = build_measurement_anchors(
            events,
            args.anchor_variable,
            lookback_hours=args.lookback_hours,
            maximum_episode_hours=args.maximum_episode_hours,
        )
        anchor_policy = {
            "kind": "observed_measurement",
            "variable": canonical_variable(args.anchor_variable),
            "lookback_hours": args.lookback_hours,
            "maximum_episode_hours": args.maximum_episode_hours,
        }
    else:
        anchors = build_regular_anchors(
            events,
            lookback_hours=args.lookback_hours,
            anchor_step_hours=args.anchor_step_hours,
            maximum_episode_hours=args.maximum_episode_hours,
        )
        anchor_policy = {
            "kind": "regular_grid",
            "lookback_hours": args.lookback_hours,
            "step_hours": args.anchor_step_hours,
            "maximum_episode_hours": args.maximum_episode_hours,
        }
    examples = build_partitioned_examples(
        events,
        anchors,
        variables,
        horizons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    examples.to_parquet(args.output, index=False)
    report = ledger_readiness_report(examples, variables)
    report.update(
        {
            "schema": "canonical_event_examples.v1",
            "anchors": int(len(anchors)),
            "horizons": list(horizons),
            "anchor_policy": anchor_policy,
            "variables": list(variables),
            "output": str(args.output.resolve()),
            "row_level_data_in_report": False,
        }
    )
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": report["output"],
                "rows": report["rows"],
                "patients": report["patients"],
                "encounters": report["encounters"],
                "strict_labels": int(
                    sum(
                        value["strict_supervision_rows"]
                        for value in report["targets"].values()
                    )
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
