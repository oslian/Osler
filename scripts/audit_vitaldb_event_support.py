#!/usr/bin/env python3
"""Measure VitalDB hypoxemia-event support before downloading raw waveforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.build_vitaldb_waveform_lockbox import _selected_anchors
from whole_body_jepa.vitaldb_lockbox import (
    NUMERIC_TRACKS,
    eligible_case_ids,
    read_vitaldb_index,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    import vitaldb

    cases = read_vitaldb_index(args.cases_index)
    tracks = read_vitaldb_index(args.tracks_index)
    selected = eligible_case_ids(cases, tracks)[: args.max_cases]
    labels = []
    supported_cases = 0
    failures = 0
    numeric_names = list(NUMERIC_TRACKS.values())
    for scanned, case_id in enumerate(selected, start=1):
        try:
            numeric = np.asarray(
                vitaldb.load_case(int(case_id), numeric_names, interval=1.0),
                dtype=float,
            )
            anchors = _selected_anchors(
                numeric,
                horizon_seconds=args.horizon_seconds,
                anchor_step_seconds=args.anchor_step_seconds,
                max_anchors=args.max_anchors_per_case,
                case_id=int(case_id),
            )
        except Exception:
            failures += 1
            continue
        if anchors:
            supported_cases += 1
            labels.extend(label for _, label, _ in anchors)
        if scanned % 50 == 0:
            print(
                json.dumps(
                    {
                        "scanned_cases": scanned,
                        "supported_cases": supported_cases,
                        "rows": len(labels),
                        "events": int(sum(labels)),
                    }
                ),
                flush=True,
            )
    report = {
        "schema": "whole_body_jepa.vitaldb_event_support.v1",
        "selected_cases": int(len(selected)),
        "supported_cases": int(supported_cases),
        "rows": int(len(labels)),
        "events": int(sum(labels)),
        "prevalence": float(np.mean(labels)) if labels else 0.0,
        "failures": int(failures),
        "horizon_seconds": int(args.horizon_seconds),
        "selection_uses_future_label": False,
        "raw_waveforms_downloaded": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-index", type=Path, required=True)
    parser.add_argument("--tracks-index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--max-anchors-per-case", type=int, default=8)
    parser.add_argument("--anchor-step-seconds", type=int, default=300)
    parser.add_argument("--horizon-seconds", type=int, default=900)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
