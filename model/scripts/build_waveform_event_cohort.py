#!/usr/bin/env python3
"""Build a leakage-safe MIMIC-IV pre-anchor multimodal hypoxemia cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from whole_body_jepa.waveform_event import (
    SustainedHypoxemiaDefinition,
    incident_sustained_hypoxemia,
    cough_effort_score,
    cough_reflex_score,
    oxygen_support_level,
    productive_cough,
    read_multiscale_waveform_features,
    respiratory_distress,
    signed_leading_score,
)


NUMERIC_CHANNELS = {
    "heart_rate": ("HR [bpm]", "btbHR [bpm]", "Pulse (SpO2) [bpm]"),
    "respiratory_rate": ("RR [rpm]",),
    "o2sat": ("SpO2 [%]",),
    "map": ("ABPm [mmHg]", "NBPm [mmHg]"),
    "sbp": ("ABPs [mmHg]", "NBPs [mmHg]"),
    "dbp": ("ABPd [mmHg]", "NBPd [mmHg]"),
}

CHARTEVENT_GROUPS = {
    "o2_device": ((226732,), 6.0, "categorical"),
    "gcs_eye": ((220739,), 12.0, "numeric"),
    "gcs_motor": ((223901,), 12.0, "numeric"),
    "gcs_verbal": ((223900,), 12.0, "numeric"),
    "fio2": ((223835, 229841), 6.0, "numeric"),
    "peep": ((220339, 224699, 224700), 6.0, "numeric"),
    "vent_mode": ((229314, 223849), 6.0, "categorical"),
    "respiratory_effort": ((223990, 229323), 6.0, "categorical"),
    "consciousness": ((226104,), 12.0, "categorical"),
    "rass": ((228096,), 4.0, "signed_score"),
    "cough_reflex": ((223905,), 6.0, "cough_reflex"),
    "cough_effort": ((223991,), 6.0, "cough_effort"),
    "productive_cough": ((223992,), 6.0, "productive_cough"),
    "dyspnea_score": ((229323,), 6.0, "signed_score"),
}

INTENT_TYPES = {
    "oxygen": ("oxygen therapy",),
    "mechanical_ventilation": ("mechanical ventilation",),
    "extubation": ("extubate",),
    "noninvasive": ("bipap", "cpap for osa"),
}

PROCEDURE_ITEMS = {
    "invasive_ventilation": (225792,),
    "noninvasive_ventilation": (225794,),
}

INPUTEVENT_GROUPS = {
    "sedative": (221385, 221668, 221712, 222168, 225150, 229420),
    "opioid": (221744, 221833, 225154, 225942, 225972),
    "neuromuscular_blocker": (221555, 222062, 229233),
    "vasopressor": (221289, 221749, 221906, 222315, 229630, 229631, 229632),
}


def _csv(path: Path) -> str:
    return f"read_csv_auto('{str(path).replace(chr(39), chr(39) * 2)}', header=true)"


def _table(root: Path, module: str, name: str) -> Path:
    for path in (root / module / f"{name}.csv.gz", root / f"{name}.csv.gz"):
        if path.exists():
            return path
    raise FileNotFoundError(f"missing MIMIC table: {module}/{name}.csv.gz")


def parse_master_header(path: Path) -> dict[str, Any] | None:
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    body = [line for line in lines if not line.startswith("#")]
    if not body or "/" not in body[0].split()[0]:
        return None
    header = body[0].split()
    if len(header) < 6:
        return None
    comments = {}
    for line in lines:
        match = re.match(r"#\s*(subject_id|hadm_id)\s+(\d+)", line)
        if match:
            comments[match.group(1)] = match.group(2)
    base = pd.to_datetime(f"{header[5]} {header[4]}", dayfirst=True, errors="coerce")
    if pd.isna(base) or "subject_id" not in comments:
        return None
    segments = []
    for line in body[1:]:
        parts = line.split()
        if len(parts) >= 2:
            segments.append((parts[0], int(float(parts[1]))))
    return {
        "record": header[0].split("/", 1)[0],
        "fs": float(header[2].split("/", 1)[0]),
        "base_datetime": base,
        "subject_id": str(comments["subject_id"]),
        "hadm_id": str(comments.get("hadm_id", "")),
        "segments": segments,
    }


def build_segment_index(waveform_root: Path, mimic_root: Path) -> pd.DataFrame:
    rows = []
    for master in sorted(waveform_root.glob("waves/*/*/*/*.hea")):
        if "_" in master.stem:
            continue
        parsed = parse_master_header(master)
        if parsed is None:
            continue
        offset = 0
        for segment, length in parsed["segments"]:
            segment_path = master.parent / segment
            if length > 0 and segment_path.with_suffix(".hea").exists():
                rows.append(
                    {
                        "subject_id": parsed["subject_id"],
                        "hadm_id": parsed["hadm_id"],
                        "record": parsed["record"],
                        "numeric_path": str(master.parent / f"{parsed['record']}n.csv.gz"),
                        "segment_path": str(segment_path),
                        "segment_start": parsed["base_datetime"]
                        + pd.to_timedelta(offset / parsed["fs"], unit="s"),
                        "segment_end": parsed["base_datetime"]
                        + pd.to_timedelta((offset + length) / parsed["fs"], unit="s"),
                        "fs": parsed["fs"],
                    }
                )
            offset += max(length, 0)
    segments = pd.DataFrame(rows)
    stays = pd.read_csv(
        _table(mimic_root, "icu", "icustays"),
        usecols=("subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"),
        dtype={"subject_id": str, "hadm_id": str, "stay_id": "int64"},
    )
    stays["intime"] = pd.to_datetime(stays["intime"])
    stays["outtime"] = pd.to_datetime(stays["outtime"])
    merged = segments.merge(stays, on=["subject_id", "hadm_id"], how="left")
    return merged[
        merged["stay_id"].notna()
        & (merged["segment_end"] >= merged["intime"])
        & (merged["segment_start"] <= merged["outtime"])
    ].reset_index(drop=True)


def _first_available(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    output = np.full(len(frame), np.nan, dtype=float)
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        output = np.where(np.isfinite(output), output, values)
    return output


def _numeric_history_features(
    times: np.ndarray,
    values: dict[str, np.ndarray],
    anchor: float,
) -> dict[str, float]:
    output = {}
    for name, array in values.items():
        for seconds in (300.0, 900.0):
            mask = (times <= anchor) & (times >= anchor - seconds) & np.isfinite(array)
            observed = array[mask]
            prefix = f"num_{name}_{int(seconds)}s"
            if not len(observed):
                output.update(
                    {
                        f"{prefix}_last": np.nan,
                        f"{prefix}_median": np.nan,
                        f"{prefix}_std": np.nan,
                        f"{prefix}_slope": np.nan,
                        f"{prefix}_count": 0.0,
                    }
                )
                continue
            observed_times = times[mask]
            slope = np.nan
            if len(observed) >= 2 and np.ptp(observed_times) > 0:
                slope = float(np.polyfit((observed_times - anchor) / 60.0, observed, 1)[0])
            output.update(
                {
                    f"{prefix}_last": float(observed[-1]),
                    f"{prefix}_median": float(np.median(observed)),
                    f"{prefix}_std": float(np.std(observed)),
                    f"{prefix}_slope": slope,
                    f"{prefix}_count": float(len(observed)),
                }
            )
    return output


def build_anchor_rows(
    segments: pd.DataFrame,
    *,
    max_per_record: int,
    max_records: int | None,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output = []
    grouped = list(segments.groupby("record", sort=True))
    if max_records is not None:
        grouped = grouped[: int(max_records)]
    definition = SustainedHypoxemiaDefinition()
    for record_index, (record, record_segments) in enumerate(grouped, start=1):
        numeric_path = Path(str(record_segments.iloc[0]["numeric_path"]))
        if not numeric_path.exists():
            continue
        numeric = pd.read_csv(numeric_path)
        if "time" not in numeric or "SpO2 [%]" not in numeric:
            continue
        times = pd.to_numeric(numeric["time"], errors="coerce").to_numpy(dtype=float) / 1000.0
        spo2 = pd.to_numeric(numeric["SpO2 [%]"], errors="coerce").to_numpy(dtype=float)
        values = {name: _first_available(numeric, columns) for name, columns in NUMERIC_CHANNELS.items()}
        observed_anchor_times = times[np.isfinite(spo2)]
        spaced = []
        last_anchor = -np.inf
        for anchor in observed_anchor_times:
            if anchor - last_anchor >= 300.0:
                spaced.append(float(anchor))
                last_anchor = float(anchor)
        rng.shuffle(spaced)
        positives = []
        negatives = []
        half = max(1, max_per_record // 2)
        for anchor in spaced:
            label, descriptor = incident_sustained_hypoxemia(times, spo2, float(anchor), definition)
            if label is None:
                continue
            parsed = parse_master_header(numeric_path.with_name(f"{record}.hea"))
            if parsed is None:
                continue
            anchor_time = parsed["base_datetime"] + pd.to_timedelta(anchor, unit="s")
            segment_rows = record_segments[
                (record_segments["segment_start"] + pd.to_timedelta(900, unit="s") <= anchor_time)
                & (record_segments["segment_end"] >= anchor_time)
            ]
            if segment_rows.empty:
                continue
            segment = segment_rows.iloc[-1]
            anchor_sample = int(round((anchor_time - segment["segment_start"]).total_seconds() * float(segment["fs"])))
            payload = {
                "subject_id": str(segment["subject_id"]),
                "hadm_id": str(segment["hadm_id"]),
                "stay_id": int(segment["stay_id"]),
                "careunit": str(segment["first_careunit"]),
                "anchor_time": anchor_time,
                "record": str(record),
                "segment_path": str(segment["segment_path"]),
                "anchor_sample": anchor_sample,
                "event": int(label),
                **descriptor,
            }
            payload["anchor_seconds"] = float(anchor)
            (positives if label else negatives).append(payload)
            if len(positives) >= half and len(negatives) >= half:
                break
        chosen = positives[:half] + negatives[:half]
        if len(chosen) < max_per_record:
            remainder = positives[half:] + negatives[half:]
            chosen += remainder[: max_per_record - len(chosen)]
        for payload in chosen:
            payload.update(
                _numeric_history_features(
                    times, values, float(payload.pop("anchor_seconds"))
                )
            )
        output.extend(chosen)
        if record_index % 20 == 0:
            print(f"anchors: {record_index}/{len(grouped)} records, {len(output)} rows", flush=True)
    return pd.DataFrame(output)


def _event_index(frame: pd.DataFrame, key: str, time: str) -> dict[Any, pd.DataFrame]:
    return {
        value: group.sort_values(time, kind="stable").reset_index(drop=True)
        for value, group in frame.groupby(key, sort=False)
    }


def _latest(
    index: dict[Any, pd.DataFrame], key: Any, anchor: pd.Timestamp, max_age_h: float
) -> pd.Series | None:
    events = index.get(key)
    if events is None or events.empty:
        return None
    times = events["event_time"].to_numpy(dtype="datetime64[ns]")
    position = int(np.searchsorted(times, np.datetime64(anchor), side="right") - 1)
    if position < 0:
        return None
    row = events.iloc[position]
    age = (anchor - row["event_time"]).total_seconds() / 3600.0
    return row if 0.0 <= age <= max_age_h else None


def attach_clinical_context(anchors: pd.DataFrame, mimic_root: Path) -> pd.DataFrame:
    stays = ",".join(str(int(value)) for value in sorted(anchors["stay_id"].unique()))
    hadms = ",".join(str(int(value)) for value in sorted(anchors["hadm_id"].astype(int).unique()))
    itemids = sorted({item for spec in CHARTEVENT_GROUPS.values() for item in spec[0]})
    con = duckdb.connect()
    chart = con.execute(
        f"""
        SELECT stay_id::BIGINT stay_id, itemid::BIGINT itemid,
               charttime event_time, "value"::VARCHAR chart_value,
               valuenum::DOUBLE valuenum
        FROM {_csv(_table(mimic_root, 'icu', 'chartevents'))}
        WHERE stay_id IN ({stays}) AND itemid IN ({','.join(map(str, itemids))})
        """
    ).df()
    chart["event_time"] = pd.to_datetime(chart["event_time"])
    chart_indices = {
        name: _event_index(chart[chart["itemid"].isin(items)], "stay_id", "event_time")
        for name, (items, _, _) in CHARTEVENT_GROUPS.items()
    }

    order_names = sorted({value for values in INTENT_TYPES.values() for value in values})
    quoted = ",".join("'" + value.replace("'", "''") + "'" for value in order_names)
    poe = con.execute(
        f"""
        SELECT hadm_id::BIGINT hadm_id, ordertime event_time,
               lower(order_subtype)::VARCHAR order_subtype,
               order_status::VARCHAR order_status
        FROM {_csv(_table(mimic_root, 'hosp', 'poe'))}
        WHERE hadm_id IN ({hadms}) AND lower(order_type) = 'respiratory'
          AND lower(order_subtype) IN ({quoted})
        """
    ).df()
    poe["event_time"] = pd.to_datetime(poe["event_time"])
    intent_indices = {
        name: _event_index(poe[poe["order_subtype"].isin(types)], "hadm_id", "event_time")
        for name, types in INTENT_TYPES.items()
    }

    procedure_items = sorted({item for values in PROCEDURE_ITEMS.values() for item in values})
    procedure = con.execute(
        f"""
        SELECT stay_id::BIGINT stay_id, itemid::BIGINT itemid,
               starttime, endtime
        FROM {_csv(_table(mimic_root, 'icu', 'procedureevents'))}
        WHERE stay_id IN ({stays}) AND itemid IN ({','.join(map(str, procedure_items))})
        """
    ).df()
    procedure["starttime"] = pd.to_datetime(procedure["starttime"])
    procedure["endtime"] = pd.to_datetime(procedure["endtime"])

    input_items = sorted({item for values in INPUTEVENT_GROUPS.values() for item in values})
    administrations = con.execute(
        f"""
        SELECT stay_id::BIGINT stay_id, itemid::BIGINT itemid,
               starttime, endtime
        FROM {_csv(_table(mimic_root, 'icu', 'inputevents'))}
        WHERE stay_id IN ({stays}) AND itemid IN ({','.join(map(str, input_items))})
          AND lower(coalesce(statusdescription, '')) <> 'rewritten'
        """
    ).df()
    administrations["starttime"] = pd.to_datetime(administrations["starttime"])
    administrations["endtime"] = pd.to_datetime(administrations["endtime"])

    rows = []
    for _, anchor_row in anchors.iterrows():
        row = anchor_row.to_dict()
        anchor = pd.Timestamp(anchor_row["anchor_time"])
        for name, (_, max_age, mode) in CHARTEVENT_GROUPS.items():
            event = _latest(chart_indices[name], int(anchor_row["stay_id"]), anchor, max_age)
            row[f"ctx_{name}_observed"] = float(event is not None)
            if event is None:
                row[f"ctx_{name}"] = np.nan
                row[f"ctx_{name}_age_h"] = np.nan
                continue
            row[f"ctx_{name}_age_h"] = (anchor - event["event_time"]).total_seconds() / 3600.0
            if mode == "numeric":
                value = float(event["valuenum"]) if pd.notna(event["valuenum"]) else np.nan
                if name == "fio2" and np.isfinite(value) and value <= 1.0:
                    value *= 100.0
            elif name == "o2_device":
                value = oxygen_support_level(event["chart_value"])
            elif name == "respiratory_effort":
                value = respiratory_distress(event["chart_value"])
            elif mode == "signed_score":
                value = signed_leading_score(event["chart_value"])
            elif mode == "cough_reflex":
                value = cough_reflex_score(event["chart_value"])
            elif mode == "cough_effort":
                value = cough_effort_score(event["chart_value"])
            elif mode == "productive_cough":
                value = productive_cough(event["chart_value"])
            else:
                value = 1.0
            row[f"ctx_{name}"] = value
        gcs = [row.get(f"ctx_gcs_{part}") for part in ("eye", "motor", "verbal")]
        row["ctx_gcs_total"] = float(np.nansum(gcs)) if np.isfinite(gcs).all() else np.nan

        for name, index in intent_indices.items():
            events = index.get(int(anchor_row["hadm_id"]))
            if events is None:
                recent = events
            else:
                recent = events[
                    (events["event_time"] <= anchor)
                    & (events["event_time"] >= anchor - pd.Timedelta(hours=6))
                ]
            row[f"intent_{name}_count_6h"] = float(0 if recent is None else len(recent))
            row[f"intent_{name}_age_h"] = (
                np.nan
                if recent is None or recent.empty
                else (anchor - recent.iloc[-1]["event_time"]).total_seconds() / 3600.0
            )

        for name, items in PROCEDURE_ITEMS.items():
            active = procedure[
                (procedure["stay_id"] == int(anchor_row["stay_id"]))
                & procedure["itemid"].isin(items)
                & (procedure["starttime"] <= anchor)
                & (procedure["endtime"].fillna(anchor) >= anchor)
            ]
            row[f"observed_{name}_active"] = float(not active.empty)

        stay_administrations = administrations[
            administrations["stay_id"] == int(anchor_row["stay_id"])
        ]
        for name, items in INPUTEVENT_GROUPS.items():
            group = stay_administrations[stay_administrations["itemid"].isin(items)]
            active = group[
                (group["starttime"] <= anchor)
                & (group["endtime"].fillna(group["starttime"]) >= anchor)
            ]
            recent_1h = group[
                (group["starttime"] <= anchor)
                & (group["starttime"] >= anchor - pd.Timedelta(hours=1))
            ]
            recent_6h = group[
                (group["starttime"] <= anchor)
                & (group["starttime"] >= anchor - pd.Timedelta(hours=6))
            ]
            row[f"tx_{name}_active"] = float(not active.empty)
            row[f"tx_{name}_starts_1h"] = float(len(recent_1h))
            row[f"tx_{name}_starts_6h"] = float(len(recent_6h))
        rows.append(row)
    return pd.DataFrame(rows)


def attach_waveforms(anchors: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows = []
    failures = 0
    for index, (_, anchor) in enumerate(anchors.iterrows(), start=1):
        try:
            features = read_multiscale_waveform_features(
                anchor["segment_path"], int(anchor["anchor_sample"])
            )
        except Exception:
            failures += 1
            continue
        row = anchor.to_dict()
        row.update(features)
        rows.append(row)
        if index % 100 == 0:
            print(f"waveforms: {index}/{len(anchors)}, failures={failures}", flush=True)
    return pd.DataFrame(rows), failures


def run(args: argparse.Namespace) -> dict[str, Any]:
    segments = build_segment_index(args.waveform_root, args.mimic_root)
    anchors = build_anchor_rows(
        segments,
        max_per_record=args.max_per_record,
        max_records=args.max_records,
        seed=args.seed,
    )
    if anchors.empty:
        raise ValueError("no eligible incident-event anchors")
    anchors = attach_clinical_context(anchors, args.mimic_root)
    cohort, failures = attach_waveforms(anchors)
    if cohort.empty:
        raise ValueError("waveform extraction produced no rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(args.output, index=False)
    feature_groups = {
        "numeric_history": [column for column in cohort if column.startswith("num_")],
        "waveform": [column for column in cohort if column.startswith("wf_") or column == "waveform_history_seconds"],
        "bedside": [column for column in cohort if column.startswith("ctx_")],
        "treatment_intent": [column for column in cohort if column.startswith("intent_")],
        "observed_intervention": [column for column in cohort if column.startswith("observed_")],
        "administered_treatment": [column for column in cohort if column.startswith("tx_")],
    }
    report = {
        "schema": "whole_body_jepa.multimodal_hypoxemia_cohort.v1",
        "event_definition": {
            "name": "incident_sustained_hypoxemia_1h",
            "threshold": 92.0,
            "current_state": "no SpO2 <= 92 in prior 10 minutes",
            "future_event": "two SpO2 <= 92 readings separated by 30 seconds to 5 minutes within 1 hour",
        },
        "rows": int(len(cohort)),
        "patients": int(cohort["subject_id"].nunique()),
        "stays": int(cohort["stay_id"].nunique()),
        "careunits": int(cohort["careunit"].nunique()),
        "event_prevalence": float(cohort["event"].mean()),
        "waveform_failures": int(failures),
        "feature_counts": {name: len(columns) for name, columns in feature_groups.items()},
        "row_level_output_committable": False,
        "causal_claim_allowed": False,
        "clinical_decision_allowed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform-root", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-per-record", type=int, default=12)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
