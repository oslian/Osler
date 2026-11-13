"""Audit MIMIC-IV support for value and abnormal-risk biomarker tasks.

This is a support audit, not a promotion audit. It reads only selected numeric
laboratory events and writes aggregate counts. It deliberately does not model
or predict next-measurement time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from osler_jepa.biomarker_prediction import (
    MIMIC_BIOMARKER_ITEMIDS,
    PLAUSIBLE_BIOMARKER_BOUNDS,
    prediction_contract,
)


MIMIC_DIR = Path(os.environ.get("MIMIC_DIR", "/Users/chunyouchang/Downloads/mimic-iv-3.1"))
MIN_PAIRS = 100
MIN_SUBJECTS = 30
MIN_CLASSES = 30


def _csv(path: Path) -> str:
    return f"read_csv_auto('{path}', all_varchar=false)"


def _item_case() -> str:
    return "CASE " + " ".join(
        f"WHEN le.itemid IN ({','.join(map(str, itemids))}) THEN '{target}'"
        for target, itemids in MIMIC_BIOMARKER_ITEMIDS.items()
    ) + " END"


def load_events(mimic_dir: Path) -> pd.DataFrame:
    import duckdb

    lab = mimic_dir / "hosp" / "labevents.csv.gz"
    stays = mimic_dir / "icu" / "icustays.csv.gz"
    if not lab.exists() or not stays.exists():
        raise FileNotFoundError(f"MIMIC-IV files not found under {mimic_dir}")
    itemids = sorted({item for values in MIMIC_BIOMARKER_ITEMIDS.values() for item in values})
    query = f"""
        SELECT ie.subject_id, ie.stay_id, ie.first_careunit,
               le.charttime, {_item_case()} AS target,
               TRY_CAST(le.valuenum AS DOUBLE) AS value,
               TRY_CAST(le.ref_range_lower AS DOUBLE) AS reference_low,
               TRY_CAST(le.ref_range_upper AS DOUBLE) AS reference_high
        FROM {_csv(lab)} le
        JOIN {_csv(stays)} ie
          ON le.hadm_id = ie.hadm_id
         AND le.charttime BETWEEN ie.intime AND ie.outtime
        WHERE le.itemid IN ({','.join(map(str, itemids))})
          AND le.valuenum IS NOT NULL
          AND le.charttime IS NOT NULL
    """
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    frame = connection.execute(query).df()
    connection.close()
    if frame.empty:
        return frame
    frame["charttime"] = pd.to_datetime(frame["charttime"], errors="coerce")
    for column in ("value", "reference_low", "reference_high"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["subject_id", "stay_id", "charttime", "target", "value"])
    for target, bounds in PLAUSIBLE_BIOMARKER_BOUNDS.items():
        mask = frame["target"].eq(target)
        frame = frame.loc[~mask | frame["value"].between(*bounds)].copy()
    return frame.sort_values(["target", "stay_id", "charttime"], kind="stable")


def target_support(frame: pd.DataFrame, target: str) -> dict[str, object]:
    subset = frame[frame["target"].eq(target)].copy()
    contract = prediction_contract(target)
    if subset.empty:
        return {
            "target": target,
            "contract": contract,
            "rows": 0,
            "value_forecast_audit_eligible": False,
            "abnormal_risk_audit_eligible": False,
            "failure_reasons": ["no_numeric_icu_events"],
        }
    subset["next_time"] = subset.groupby("stay_id", sort=False)["charttime"].shift(-1)
    subset["next_value"] = subset.groupby("stay_id", sort=False)["value"].shift(-1)
    subset["next_low"] = subset.groupby("stay_id", sort=False)["reference_low"].shift(-1)
    subset["next_high"] = subset.groupby("stay_id", sort=False)["reference_high"].shift(-1)
    subset["elapsed_hr"] = (subset["next_time"] - subset["charttime"]).dt.total_seconds() / 3600.0
    future = subset[subset["elapsed_hr"].between(0.5, 72.0, inclusive="both")].copy()
    bounded = future[
        future["next_value"].notna()
        & (future["next_low"].notna() | future["next_high"].notna())
    ].copy()
    bounded["future_abnormal"] = (
        (bounded["next_low"].notna() & bounded["next_value"].lt(bounded["next_low"]))
        | (bounded["next_high"].notna() & bounded["next_value"].gt(bounded["next_high"]))
    )
    positives = int(bounded["future_abnormal"].sum())
    negatives = int((~bounded["future_abnormal"]).sum())
    value_eligible = bool(
        contract["numeric_future_candidate"]
        and len(future) >= MIN_PAIRS
        and future["subject_id"].nunique() >= MIN_SUBJECTS
    )
    risk_eligible = bool(
        contract["future_abnormal_risk_candidate"]
        and len(bounded) >= MIN_PAIRS
        and bounded["subject_id"].nunique() >= MIN_SUBJECTS
        and positives >= MIN_CLASSES
        and negatives >= MIN_CLASSES
    )
    reasons = []
    if contract["numeric_future_candidate"] and not value_eligible:
        reasons.append("insufficient_repeated_future_value_support")
    if contract["future_abnormal_risk_candidate"] and not risk_eligible:
        reasons.append("insufficient_reference_range_or_class_support")
    return {
        "target": target,
        "contract": contract,
        "rows": int(len(subset)),
        "subjects": int(subset["subject_id"].nunique()),
        "stays": int(subset["stay_id"].nunique()),
        "repeated_future_pairs_0_5_to_72h": int(len(future)),
        "future_pair_subjects": int(future["subject_id"].nunique()),
        "reference_labeled_pairs": int(len(bounded)),
        "future_abnormal_positive": positives,
        "future_abnormal_negative": negatives,
        "value_forecast_audit_eligible": value_eligible,
        "abnormal_risk_audit_eligible": risk_eligible,
        "failure_reasons": reasons,
    }


def run(mimic_dir: Path) -> dict[str, object]:
    events = load_events(mimic_dir)
    results = [target_support(events, target) for target in MIMIC_BIOMARKER_ITEMIDS]
    return {
        "schema": "mimiciv_biomarker_prediction_support.v1",
        "database": "MIMIC-IV v3.1",
        "results": results,
        "value_forecast_audit_eligible": sorted(
            item["target"] for item in results if item["value_forecast_audit_eligible"]
        ),
        "abnormal_risk_audit_eligible": sorted(
            item["target"] for item in results if item["abnormal_risk_audit_eligible"]
        ),
        "policy": {
            "next_measurement_time_modeled": False,
            "support_does_not_equal_validation": True,
            "future_information_used_as_feature": False,
            "causal_claim_allowed": False,
            "row_level_data_written": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.mimic_dir)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "value_forecast_audit_eligible": report["value_forecast_audit_eligible"],
        "abnormal_risk_audit_eligible": report["abnormal_risk_audit_eligible"],
        "next_measurement_time_modeled": False,
    }, indent=2))


if __name__ == "__main__":
    main()
