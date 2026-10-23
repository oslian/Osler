"""Audit long-horizon prediction of sparse MIMIC-IV laboratory targets.

This script reads the raw MIMIC-IV labevents table instead of relying on the
short 6-hour observation parquet.  It builds aggregate-only anchor/future
pairs at 6/12/24/48 hours for sparse hormones, tumor/cardiac markers,
coagulation tests, and related labs.

The candidate is a factual ridge residual model using only observations at or
before the anchor.  It is compared with persistence and a capacity-matched
baseline is not claimed to be causal.  A target is promoted only if the
candidate beats persistence on all seven patient-heldout seeds and on a
care-unit heldout split.  MIMIC-IV is single-center, so the care-unit split is
reported as an internal transport check, not an external hospital test.

No row-level values or identifiers are written to the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_target_router import _bootstrap_ci, _feature_columns, _fit_ridge, _round, split_subjects
from osler_jepa.biomarker_prediction import (
    MIMIC_BIOMARKER_ITEMIDS,
    PLAUSIBLE_BIOMARKER_BOUNDS,
    registered_numeric_targets,
)


PROJECT = Path(__file__).resolve().parent
MIMIC_DIR = Path(os.environ.get("MIMIC_DIR", "/Users/chunyouchang/Downloads/mimic-iv-3.1"))
SEEDS = (7, 11, 19, 23, 37, 53, 71)
HORIZONS = (6, 12, 24, 48)


def _time_ns(values: pd.Series | pd.Index) -> np.ndarray:
    """Return timestamps as int64 nanoseconds across pandas time units."""

    return pd.to_datetime(values).astype("datetime64[ns]").astype("int64").to_numpy(dtype=np.int64)

ITEMIDS: dict[str, tuple[int, ...]] = {
    "troponin_i": (51002, 52642),
    "troponin_t": (51003,),
    "bnp": (50963,),
    "ck_mb": (50911, 51580),
    "cpk": (50910,),
    "inr": (51237, 51675),
    "ptt": (51275, 52923),
    "fibrinogen": (51214, 51623, 52116),
    "tsh": (50993,),
    "total_t4": (50994,),
    "free_t4": (50995,),
    "cortisol": (50909,),
    "pth": (50965,),
    "tpo_antibody": (50992,),
    "crp": (50889,),
    "ferritin": (50924,),
    "bilirubin": (50885, 53089),
    "bilirubin_direct": (50883,),
    "albumin": (50862, 53085),
    "hemoglobin": (50811, 51222, 51640),
    "hematocrit": (50810, 51221, 51638, 51639, 52028),
    "creatinine": (50912, 52024, 52546),
    "bun": (51006, 52647),
    "bicarbonate": (50803, 50882),
    "anion_gap": (50868, 52500),
    "sodium": (50824, 50983, 52455, 52623),
    "potassium": (50822, 50971, 52452, 52610),
    "glucose": (50809, 50931, 52027, 52569),
    **MIMIC_BIOMARKER_ITEMIDS,
}

TARGETS = tuple(dict.fromkeys((
    "cortisol", "pth", "tpo_antibody", "tsh", "total_t4", "free_t4",
    "troponin_i", "troponin_t", "bnp", "ck_mb", "cpk", "inr", "ptt",
    "fibrinogen", "crp", "ferritin", "bilirubin", "bilirubin_direct",
    *registered_numeric_targets(),
)))

PLAUSIBLE: dict[str, tuple[float, float]] = {
    "cortisol": (0.0, 300.0),
    "pth": (0.0, 3000.0),
    "tpo_antibody": (0.0, 5000.0),
    "tsh": (0.0, 100.0),
    "total_t4": (0.0, 30.0),
    "free_t4": (0.0, 20.0),
    "troponin_i": (0.0, 100000.0),
    "troponin_t": (0.0, 100000.0),
    "bnp": (0.0, 100000.0),
    "ck_mb": (0.0, 100000.0),
    "cpk": (0.0, 1000000.0),
    "inr": (0.0, 20.0),
    "ptt": (0.0, 300.0),
    "fibrinogen": (0.0, 2000.0),
    "crp": (0.0, 1000.0),
    "ferritin": (0.0, 100000.0),
    "bilirubin": (0.0, 100.0),
    "bilirubin_direct": (0.0, 100.0),
    "albumin": (0.0, 10.0),
    "hemoglobin": (0.0, 30.0),
    "hematocrit": (0.0, 100.0),
    "creatinine": (0.0, 30.0),
    "bun": (0.0, 300.0),
    "bicarbonate": (0.0, 80.0),
    "anion_gap": (0.0, 100.0),
    "sodium": (80.0, 220.0),
    "potassium": (0.0, 15.0),
    "glucose": (0.0, 2000.0),
    **PLAUSIBLE_BIOMARKER_BOUNDS,
}


def _csv(path: Path) -> str:
    return f"read_csv_auto('{path}', all_varchar=false)"


def _item_case() -> str:
    parts = []
    for target, itemids in ITEMIDS.items():
        parts.append(f"WHEN le.itemid IN ({','.join(str(x) for x in itemids)}) THEN '{target}'")
    return "CASE " + " ".join(parts) + " END"


def load_events(mimic_dir: Path) -> pd.DataFrame:
    """Scan only selected itemids from raw labevents and attach ICU metadata."""

    import duckdb

    itemids = sorted({itemid for values in ITEMIDS.values() for itemid in values})
    lab_path = mimic_dir / "hosp" / "labevents.csv.gz"
    icu_path = mimic_dir / "icu" / "icustays.csv.gz"
    if not lab_path.exists() or not icu_path.exists():
        raise FileNotFoundError(f"MIMIC files not found under {mimic_dir}")
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    query = f"""
        SELECT ie.subject_id, ie.stay_id, ie.first_careunit,
               ie.intime, ie.outtime, le.charttime,
               {_item_case()} AS target,
               TRY_CAST(le.valuenum AS DOUBLE) AS value
        FROM {_csv(lab_path)} AS le
        JOIN {_csv(icu_path)} AS ie
          ON le.hadm_id = ie.hadm_id
         AND le.charttime BETWEEN ie.intime AND ie.outtime
        WHERE le.itemid IN ({','.join(str(x) for x in itemids)})
          AND le.valuenum IS NOT NULL
          AND le.charttime IS NOT NULL
    """
    print("Scanning selected MIMIC-IV labevents itemids", flush=True)
    frame = connection.execute(query).df()
    connection.close()
    if frame.empty:
        return frame
    frame["charttime"] = pd.to_datetime(frame["charttime"])
    frame["intime"] = pd.to_datetime(frame["intime"])
    frame["outtime"] = pd.to_datetime(frame["outtime"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["target"] = frame["target"].astype(str)
    frame = frame.dropna(subset=["subject_id", "stay_id", "charttime", "target", "value"]).copy()
    frame["hours_since_intime"] = (frame["charttime"] - frame["intime"]).dt.total_seconds() / 3600.0
    for target, (low, high) in PLAUSIBLE.items():
        mask = frame["target"].eq(target)
        frame.loc[mask, "value"] = frame.loc[mask, "value"].where(
            frame.loc[mask, "value"].between(low, high)
        )
    frame = frame.dropna(subset=["value"]).sort_values(["stay_id", "charttime", "target"]).reset_index(drop=True)
    return frame


def _event_arrays(events: pd.DataFrame) -> dict[tuple[object, str], tuple[np.ndarray, np.ndarray]]:
    arrays = {}
    for (stay_id, target), group in events.groupby(["stay_id", "target"], sort=False):
        ordered = group.sort_values("charttime")
        arrays[(stay_id, target)] = (
            _time_ns(ordered["charttime"]),
            ordered["value"].to_numpy(dtype=np.float64),
        )
    return arrays


def build_target_anchors(
    events: pd.DataFrame,
    target: str,
    horizons: tuple[int, ...],
    arrays: dict[tuple[object, str], tuple[np.ndarray, np.ndarray]],
    *,
    endpoint: bool = False,
) -> pd.DataFrame:
    """Build one anchor table and reuse it for every requested horizon.

    ``endpoint=True`` selects the first observation in the 0.75h--1.25h
    endpoint window.  The default preserves the original "next observation
    within horizon" audit used for support accounting.
    """

    if events.empty:
        return pd.DataFrame()
    target_events = events[events["target"].eq(target)].copy()
    rows = []
    for stay_id, group in target_events.groupby("stay_id", sort=False):
        ordered = group.sort_values("charttime")
        times = _time_ns(ordered["charttime"])
        values = ordered["value"].to_numpy(dtype=np.float64)
        for anchor_ns, current in zip(times, values):
            anchor = pd.Timestamp(anchor_ns)
            row = {
                "subject_id": ordered["subject_id"].iloc[0],
                "stay_id": stay_id,
                "first_careunit": ordered["first_careunit"].iloc[0],
                "anchor_time": anchor,
                "hours_since_intime": float((anchor - ordered["intime"].iloc[0]).total_seconds() / 3600.0),
                "target": target,
                f"{target}_t": float(current),
            }
            for horizon_hours in horizons:
                minimum_age = max(0.5, 0.75 * horizon_hours) if endpoint else 0.5
                maximum_age = 1.25 * horizon_hours if endpoint else float(horizon_hours)
                minimum = anchor_ns + int(minimum_age * 3600 * 1_000_000_000)
                maximum = anchor_ns + int(maximum_age * 3600 * 1_000_000_000)
                index = int(np.searchsorted(times, minimum, side="left"))
                if index < len(times) and times[index] <= maximum:
                    row[f"{target}_tp{horizon_hours}"] = float(values[index])
                    row[f"{target}_future_age_hr_{horizon_hours}"] = float((times[index] - anchor_ns) / 3.6e12)
                else:
                    row[f"{target}_tp{horizon_hours}"] = np.nan
                    row[f"{target}_future_age_hr_{horizon_hours}"] = np.nan
            rows.append(row)
    anchors = pd.DataFrame(rows)
    if anchors.empty:
        return anchors

    # Add only anchor-time state values.  Each value is the latest observation
    # at or before the anchor; future observations never enter the features.
    source_targets = tuple(sorted(ITEMIDS))
    for source in source_targets:
        current_column = f"{source}_t"
        values = np.full(len(anchors), np.nan, dtype=np.float64)
        ages = np.full(len(anchors), np.nan, dtype=np.float64)
        for positions, (stay_id, group) in enumerate(anchors.groupby("stay_id", sort=False)):
            event = arrays.get((stay_id, source))
            if event is None:
                continue
            event_times, event_values = event
            indices = group.index.to_numpy(dtype=np.int64)
            anchor_times = group["anchor_time"].astype("int64").to_numpy(dtype=np.int64)
            selected = np.searchsorted(event_times, anchor_times, side="right") - 1
            valid = selected >= 0
            if valid.any():
                values[indices[valid]] = event_values[selected[valid]]
                ages[indices[valid]] = (anchor_times[valid] - event_times[selected[valid]]) / 3.6e12
        anchors[current_column] = values
        anchors[f"{source}_age_hr"] = ages
    return anchors.reset_index(drop=True)


def _delta_by_subject(rows: pd.DataFrame, truth: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> pd.Series:
    mask = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(baseline)
    if not mask.any():
        return pd.Series(dtype="float64")
    temp = rows.loc[mask, ["subject_id"]].copy()
    temp["delta"] = np.abs(pred[mask] - truth[mask]) - np.abs(baseline[mask] - truth[mask])
    return temp.groupby("subject_id")["delta"].mean()


def _summary(rows: pd.DataFrame, truth: np.ndarray, pred: np.ndarray, baseline: np.ndarray, seed: int) -> dict[str, object]:
    values = _delta_by_subject(rows, truth, pred, baseline).to_numpy(dtype=np.float64)
    if len(values) == 0:
        return {"subjects": 0, "point_delta": None, "bootstrap_95_ci": [None, None], "significant": False}
    ci = _bootstrap_ci(values, seed=seed, samples=300)
    point = float(values.mean())
    return {
        "subjects": int(len(values)),
        "point_delta": _round(point),
        "bootstrap_95_ci": ci,
        "beats_persistence": bool(point < 0.0),
        "significant": bool(ci is not None and ci[1] < 0.0),
    }


def _split_report(frame: pd.DataFrame, target: str, horizon: int, seed: int, group_column: str = "subject_id") -> dict[str, object]:
    future_suffix = f"tp{horizon}"
    discovery_groups, heldout_groups = split_subjects(frame, seed=seed, discovery_fraction=0.67, group_column=group_column)
    train = frame[frame[group_column].isin(discovery_groups)].copy()
    test = frame[frame[group_column].isin(heldout_groups)].copy()
    current = f"{target}_t"
    future = f"{target}_{future_suffix}"
    if train.empty or test.empty or current not in train or future not in train:
        return {"status": "insufficient_pairs", "rows": int(len(test)), "subjects": 0}
    features = [column for column in _feature_columns(train, target) if column != future]
    # Treat the sparse target's current value as the persistence anchor, while
    # allowing the model to use other measurements and their ages.
    candidate = _fit_ridge(train, test, target, features, 10.0, future_suffix)
    truth = test[future].to_numpy(dtype=np.float64)
    baseline = test[current].to_numpy(dtype=np.float64)
    return {
        "status": "evaluated",
        "rows": int(len(test)),
        "subjects": int(test["subject_id"].nunique()),
        "feature_count": int(len(features)),
        "candidate_vs_persistence": _summary(test, truth, candidate, baseline, seed + 11),
    }


def evaluate_pairs(frame: pd.DataFrame, target: str, horizon: int) -> dict[str, object]:
    if frame.empty:
        return {"status": "no_pairs", "target": target, "horizon_hours": horizon, "pairs": 0}
    patient_reports = [_split_report(frame, target, horizon, seed) for seed in SEEDS]
    care_report = None
    if frame["first_careunit"].nunique(dropna=True) >= 2:
        care_report = _split_report(frame, target, horizon, seed=9901, group_column="first_careunit")
    patient_pass = all(
        report.get("status") == "evaluated"
        and report.get("candidate_vs_persistence", {}).get("significant")
        for report in patient_reports
    )
    care_pass = bool(care_report and care_report.get("candidate_vs_persistence", {}).get("significant"))
    return {
        "status": "evaluated",
        "target": target,
        "horizon_hours": horizon,
        "pairs": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "care_units": int(frame["first_careunit"].nunique()),
        "patient_splits": patient_reports,
        "care_unit_holdout": care_report,
        "promotion_status": "validated" if patient_pass and care_pass else "candidate_or_fallback",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/mimiciv_sparse_lab_long_horizon_audit.json"))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--horizons", default=",".join(str(value) for value in HORIZONS))
    args = parser.parse_args()
    events = load_events(args.mimic_dir)
    target_names = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    horizons = tuple(int(item.strip()) for item in args.horizons.split(",") if item.strip())
    results = []
    print("Building one reusable event index for all sparse targets/horizons", flush=True)
    arrays = _event_arrays(events)
    for target in target_names:
        available = int(events.loc[events["target"].eq(target), "value"].notna().sum()) if not events.empty else 0
        print(f"{target}: raw_observations={available}", flush=True)
        anchors = build_target_anchors(events, target, horizons, arrays)
        for horizon in horizons:
            future = f"{target}_tp{horizon}"
            pairs = anchors[anchors[future].notna()].copy() if not anchors.empty else anchors
            report = evaluate_pairs(pairs, target, horizon)
            results.append(report)
            print(f"  {horizon}h pairs={report.get('pairs', 0)} status={report.get('promotion_status', report.get('status'))}", flush=True)
    validated = [
        {"target": item["target"], "horizon_hours": item["horizon_hours"], "pairs": item["pairs"]}
        for item in results if item.get("promotion_status") == "validated"
    ]
    output = {
        "artifact": "MIMIC-IV sparse laboratory long-horizon factual audit",
        "mimic_dir": str(args.mimic_dir),
        "raw_rows": int(len(events)),
        "raw_subjects": int(events["subject_id"].nunique()) if not events.empty else 0,
        "raw_stays": int(events["stay_id"].nunique()) if not events.empty else 0,
        "targets": list(target_names),
        "horizons_hours": list(horizons),
        "validated_cells": validated,
        "results": results,
        "boundary": {
            "factual_observation_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "mimic_single_center_external_hospital_claim_allowed": False,
            "row_level_outputs_written": False,
            "sparse_target_without_pairs_falls_back": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "validated_cells": validated}, indent=2))


if __name__ == "__main__":
    main()
