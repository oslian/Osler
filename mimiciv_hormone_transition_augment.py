"""Augment an existing MIMIC-IV transition cohort with hormone lab targets.

The full observation cohort already contains aligned ICU anchors and a large
set of physiology targets. Rebuilding every raw table just to add a few sparse
hormone labs is unnecessarily expensive, so this adapter scans labevents once
for the hormone itemids and aligns those measurements to the existing
``t``/``t_plus`` anchors. It is observational only and grants no hidden-state,
causal, counterfactual, clinical, or runtime authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimiciv_observation_transition_extract import (
    LAB_ITEMIDS,
    clean_measurements,
    csv_expr,
    icu_path,
    hosp_path,
)


HORMONE_TARGETS = (
    "cortisol",
    "pth",
    "tpo_antibody",
    "tsh",
    "total_t4",
    "free_t4",
)


def query_hormone_measurements(mimic_dir: Path) -> pd.DataFrame:
    import duckdb

    itemids = sorted({itemid for target in HORMONE_TARGETS for itemid in LAB_ITEMIDS[target]})
    case_parts = [
        f"WHEN le.itemid IN ({','.join(str(itemid) for itemid in LAB_ITEMIDS[target])}) THEN '{target}'"
        for target in HORMONE_TARGETS
    ]
    query = f"""
        SELECT ie.subject_id, ie.stay_id, le.charttime,
               CASE {' '.join(case_parts)} END AS var,
               le.valuenum
        FROM {csv_expr(hosp_path(mimic_dir, 'labevents'))} le
        JOIN {csv_expr(icu_path(mimic_dir, 'icustays'))} ie
          ON le.hadm_id = ie.hadm_id
         AND le.charttime BETWEEN ie.intime AND ie.outtime
        WHERE le.itemid IN ({','.join(str(itemid) for itemid in itemids)})
          AND le.valuenum IS NOT NULL
    """
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    measurements = connection.execute(query).df()
    connection.close()
    if measurements.empty:
        return measurements
    return clean_measurements(measurements)


def align_target(
    anchors: pd.DataFrame,
    measurements: pd.DataFrame,
    target: str,
    when_col: str,
    direction: str,
    tolerance_hours: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    left_base = anchors[["stay_id", when_col]].copy()
    left_base[when_col] = pd.to_datetime(left_base[when_col]).astype("datetime64[ns]")
    left_sorted = left_base.sort_values(when_col)
    values = measurements.loc[
        measurements["var"] == target,
        ["stay_id", "charttime", "valuenum"],
    ].copy()
    if values.empty:
        missing = np.full(len(anchors), np.nan, dtype=np.float64)
        age = np.full(len(anchors), np.nan, dtype=np.float64) if when_col == "t" else None
        return missing, age
    values["charttime"] = pd.to_datetime(values["charttime"]).astype("datetime64[ns]")
    merged = pd.merge_asof(
        left_sorted,
        values.sort_values("charttime"),
        left_on=when_col,
        right_on="charttime",
        by="stay_id",
        direction=direction,
        tolerance=pd.Timedelta(hours=float(tolerance_hours)),
    )
    merged = merged.set_index(left_sorted.index).reindex(left_base.index)
    values_out = merged["valuenum"].to_numpy(dtype=np.float64)
    if when_col != "t":
        return values_out, None
    ages = (merged[when_col] - merged["charttime"]).dt.total_seconds() / 3600.0
    return values_out, ages.to_numpy(dtype=np.float64)


def augment(
    base_path: Path,
    mimic_dir: Path,
    output_path: Path,
    report_path: Path,
    tolerance_hours: float,
) -> dict[str, object]:
    base = pd.read_parquet(base_path)
    anchors = base[["stay_id", "t", "t_plus"]].copy()
    measurements = query_hormone_measurements(mimic_dir)
    result = base.copy()
    support: dict[str, object] = {}
    for target in HORMONE_TARGETS:
        current, age = align_target(
            anchors,
            measurements,
            target,
            "t",
            "backward",
            tolerance_hours,
        )
        future, _ = align_target(
            anchors,
            measurements,
            target,
            "t_plus",
            "nearest",
            tolerance_hours,
        )
        result[f"{target}_t"] = current
        result[f"{target}_age_hr"] = age
        result[f"{target}_tp6"] = future
        support[target] = {
            "raw_measurement_rows": int((measurements["var"] == target).sum()) if not measurements.empty else 0,
            "current_pairs": int(np.isfinite(current).sum()),
            "future_pairs": int(np.isfinite(future).sum()),
            "current_and_future_pairs": int(np.isfinite(current).astype(bool).astype(int).dot(np.isfinite(future).astype(bool).astype(int))),
        }
    result.to_parquet(output_path, index=False)
    report = {
        "artifact": "MIMIC-IV hormone target transition augmentation",
        "base_cohort": str(base_path),
        "output": str(output_path),
        "rows": int(len(result)),
        "subjects": int(result["subject_id"].nunique()) if "subject_id" in result else None,
        "stays": int(result["stay_id"].nunique()) if "stay_id" in result else None,
        "horizon_hours": 6,
        "alignment": {
            "current_direction": "backward",
            "future_direction": "nearest",
            "tolerance_hours": float(tolerance_hours),
        },
        "targets": list(HORMONE_TARGETS),
        "support": support,
        "safety_boundary": {
            "row_level_outputs_committed": False,
            "patient_ids_included_in_report": False,
            "direct_hormone_concentration_claim_allowed": False,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tolerance-hours", type=float, default=2.0)
    args = parser.parse_args()
    report = augment(
        args.base,
        args.mimic_dir,
        args.output,
        args.report,
        args.tolerance_hours,
    )
    print(json.dumps({
        "output": report["output"],
        "rows": report["rows"],
        "support": report["support"],
        "causal_claim_allowed": report["safety_boundary"]["causal_claim_allowed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
