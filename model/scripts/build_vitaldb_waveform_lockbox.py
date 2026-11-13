#!/usr/bin/env python3
"""Build an aggregate-safe independent VitalDB waveform event lockbox."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from whole_body_jepa.raw_waveform import build_raw_waveform_tensor
from whole_body_jepa.vitaldb_lockbox import (
    NUMERIC_TRACKS,
    RAW_TRACKS,
    eligible_case_ids,
    last_observation,
    numeric_history_features,
    read_vitaldb_index,
)
from whole_body_jepa.waveform_event import (
    SustainedHypoxemiaDefinition,
    incident_sustained_hypoxemia,
)


def _selected_anchors(
    numeric: np.ndarray,
    *,
    horizon_seconds: int,
    anchor_step_seconds: int,
    max_anchors: int,
    case_id: int,
) -> list[tuple[int, int, dict[str, float]]]:
    definition = SustainedHypoxemiaDefinition(horizon_seconds=float(horizon_seconds))
    times = np.arange(len(numeric), dtype=float)
    spo2 = numeric[:, list(NUMERIC_TRACKS).index("o2sat")]
    candidates = []
    for anchor in range(900, len(numeric) - horizon_seconds, anchor_step_seconds):
        label, descriptor = incident_sustained_hypoxemia(
            times, spo2, float(anchor), definition
        )
        if label is not None:
            candidates.append((anchor, int(label), descriptor))
    if len(candidates) <= max_anchors:
        return candidates
    rng = np.random.default_rng(int(case_id) + 20260827)
    selected = np.sort(rng.choice(len(candidates), size=max_anchors, replace=False))
    return [candidates[index] for index in selected]


def _row(
    case: pd.Series,
    numeric: np.ndarray,
    anchor: int,
    label: int,
    descriptor: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "subject_id": str(case["subjectid"]),
        "case_id": int(case["caseid"]),
        "careunit": str(case.get("department", "unknown")),
        "anchor_seconds": float(anchor),
        "event": int(label),
        **descriptor,
        **numeric_history_features(numeric, float(anchor)),
    }
    fio2_column = list(NUMERIC_TRACKS).index("fio2")
    peep_column = list(NUMERIC_TRACKS).index("peep")
    row.update(
        {
            "ctx_fio2": last_observation(numeric, fio2_column, float(anchor)),
            "ctx_peep": last_observation(numeric, peep_column, float(anchor)),
            "ctx_o2_device": 4.0,
            "ctx_gcs_eye": np.nan,
            "ctx_gcs_motor": np.nan,
            "ctx_gcs_total": np.nan,
            "ctx_gcs_verbal": np.nan,
            "ctx_respiratory_effort": np.nan,
            "ctx_vent_mode": np.nan,
            "intent_extubation_count_6h": np.nan,
            "intent_mechanical_ventilation_count_6h": np.nan,
            "intent_noninvasive_count_6h": np.nan,
            "intent_oxygen_count_6h": np.nan,
            "observed_invasive_ventilation_active": 1.0,
            "observed_noninvasive_ventilation_active": 0.0,
        }
    )
    return row


def _load_case_with_retry(
    vitaldb_module: object,
    case_id: int,
    track_names: list[str],
    interval: float,
    *,
    attempts: int = 4,
) -> np.ndarray:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return np.asarray(
                vitaldb_module.load_case(case_id, track_names, interval=interval),
                dtype=float,
            )
        except Exception as exception:
            error = exception
            if attempt + 1 < attempts:
                time.sleep(2.0 ** attempt)
    assert error is not None
    raise error


def _resume_rows_and_tensors(
    cohort_path: Path, raw_path: Path, enabled: bool
) -> tuple[list[dict[str, object]], list[dict[str, np.ndarray]]]:
    if not enabled or not cohort_path.exists() or not raw_path.exists():
        return [], []
    frame = pd.read_parquet(cohort_path)
    cache = np.load(raw_path)
    if len(frame) != len(cache["past"]):
        raise ValueError("resume cohort and raw cache have different row counts")
    tensors = [
        {
            "past": cache["past"][index],
            "future": cache["future"][index],
            "past_mask": cache["past_mask"][index],
            "future_mask": cache["future_mask"][index],
            "quality": cache["quality"][index],
        }
        for index in range(len(frame))
    ]
    return frame.to_dict("records"), tensors


def run(args: argparse.Namespace) -> dict[str, object]:
    import vitaldb

    cases = read_vitaldb_index(args.cases_index)
    tracks = read_vitaldb_index(args.tracks_index)
    case_lookup = cases.set_index("caseid", drop=False)
    selected_cases = eligible_case_ids(cases, tracks)[: args.max_cases]
    rows, tensors = _resume_rows_and_tensors(
        args.cohort_output, args.raw_output, args.resume
    )
    completed_cases = {int(row["case_id"]) for row in rows}
    failures: dict[str, int] = {"numeric": 0, "raw": 0, "no_anchor": 0}
    failure_reasons: Counter[str] = Counter()
    numeric_names = list(NUMERIC_TRACKS.values())
    raw_names = list(RAW_TRACKS.values())
    raw_labels = list(RAW_TRACKS)
    for scanned, case_id in enumerate(selected_cases, start=1):
        if int(case_id) in completed_cases:
            continue
        try:
            numeric = _load_case_with_retry(
                vitaldb, int(case_id), numeric_names, 1.0
            )
        except Exception as exception:
            failures["numeric"] += 1
            failure_reasons[f"numeric:{type(exception).__name__}"] += 1
            continue
        anchors = _selected_anchors(
            numeric,
            horizon_seconds=args.horizon_seconds,
            anchor_step_seconds=args.anchor_step_seconds,
            max_anchors=args.max_anchors_per_case,
            case_id=int(case_id),
        )
        if not anchors:
            failures["no_anchor"] += 1
            continue
        try:
            raw = _load_case_with_retry(
                vitaldb,
                int(case_id),
                raw_names,
                args.raw_interval_seconds,
            )
        except Exception as exception:
            failures["raw"] += 1
            failure_reasons[f"raw:{type(exception).__name__}"] += 1
            continue
        case = case_lookup.loc[int(case_id)]
        for anchor, label, descriptor in anchors:
            try:
                tensor = build_raw_waveform_tensor(
                    raw,
                    raw_labels,
                    1.0 / args.raw_interval_seconds,
                    int(round(anchor / args.raw_interval_seconds)),
                    points_per_scale=args.points_per_scale,
                    future_seconds=args.future_waveform_seconds,
                )
            except ValueError:
                failures["raw"] += 1
                continue
            rows.append(_row(case, numeric, anchor, label, descriptor))
            tensors.append(tensor)
        if scanned % 10 == 0:
            print(
                json.dumps(
                    {
                        "scanned_cases": scanned,
                        "cached_patients": len({row["subject_id"] for row in rows}),
                        "cached_rows": len(rows),
                        "events": int(sum(int(row["event"]) for row in rows)),
                    }
                ),
                flush=True,
            )
    if not rows:
        raise RuntimeError("VitalDB lockbox produced no eligible rows")
    frame = pd.DataFrame(rows)
    args.cohort_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.cohort_output, index=False)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.raw_output,
        row_index=np.arange(len(rows), dtype=np.int64),
        past=np.stack([item["past"] for item in tensors]),
        future=np.stack([item["future"] for item in tensors]),
        past_mask=np.stack([item["past_mask"] for item in tensors]),
        future_mask=np.stack([item["future_mask"] for item in tensors]),
        quality=np.stack([item["quality"] for item in tensors]),
    )
    report = {
        "schema": "whole_body_jepa.vitaldb_waveform_lockbox.v1",
        "source": "VitalDB open dataset",
        "selected_cases": int(len(selected_cases)),
        "patients": int(frame["subject_id"].nunique()),
        "rows": int(len(frame)),
        "events": int(frame["event"].sum()),
        "prevalence": float(frame["event"].mean()),
        "departments": frame.groupby("careunit")["subject_id"].nunique().astype(int).to_dict(),
        "failures": failures,
        "failure_reasons": dict(failure_reasons),
        "resumed": bool(args.resume),
        "horizon_seconds": int(args.horizon_seconds),
        "raw_sampling_hz": float(1.0 / args.raw_interval_seconds),
        "selection_uses_future_label": False,
        "row_level_outputs_committable": False,
        "causal_claim_allowed": False,
        "clinical_decision_allowed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-index", type=Path, required=True)
    parser.add_argument("--tracks-index", type=Path, required=True)
    parser.add_argument("--cohort-output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=320)
    parser.add_argument("--max-anchors-per-case", type=int, default=8)
    parser.add_argument("--anchor-step-seconds", type=int, default=300)
    parser.add_argument("--horizon-seconds", type=int, default=900)
    parser.add_argument("--raw-interval-seconds", type=float, default=0.01)
    parser.add_argument("--points-per-scale", type=int, default=256)
    parser.add_argument("--future-waveform-seconds", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
