"""Run target-specific change-hurdle rescue on selected MIMIC runtime cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from target_specific_domain_conformal_audit import (
    DEFAULT_HYPOXEMIA_THRESHOLD,
    TARGET_CHANGE_THRESHOLDS,
    _audit_horizon,
)


RESCUE_CELLS = (
    ("glucose", 1),
    ("platelets", 1),
    ("urine_output", 1),
    ("platelets", 3),
    ("creatinine", 6),
    ("bicarbonate", 12),
    ("creatinine", 12),
    ("bicarbonate", 24),
    ("bun", 24),
    ("creatinine", 24),
    ("creatinine", 48),
    ("hemoglobin", 3),
    ("urine_output", 3),
    ("hemoglobin", 6),
    ("urine_output", 6),
    ("chloride", 12),
    ("hematocrit", 12),
    ("sodium", 24),
    ("glucose", 48),
)


def cell_frame(cohort: Path, target: str, horizon: int) -> pd.DataFrame:
    frame = pd.read_parquet(cohort).copy()
    future_column = f"{target}_tp{int(horizon)}"
    required = {"subject_id", "stay_id", "t", f"{target}_t", future_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing MIMIC hurdle columns: {missing}")
    frame["anchor_time"] = pd.to_datetime(frame["t"], errors="coerce")
    frame[f"future_{target}"] = pd.to_numeric(frame[future_column], errors="coerce")
    frame["horizon_hours"] = int(horizon)
    frame["hospitalid"] = "MIMIC-IV"
    frame["careunit"] = frame.get("first_careunit", "unknown").fillna("unknown")
    return frame[
        frame["anchor_time"].notna()
        & pd.to_numeric(frame[f"{target}_t"], errors="coerce").notna()
        & frame[f"future_{target}"].notna()
    ].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--cohort-6h", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows-per-cell", type=int, default=80_000)
    parser.add_argument(
        "--cells",
        default="",
        help="Optional comma-separated target@horizon list; defaults to every rescue cell.",
    )
    args = parser.parse_args()

    requested = {
        value.strip() for value in args.cells.split(",") if value.strip()
    }
    selected_cells = tuple(
        (target, horizon)
        for target, horizon in RESCUE_CELLS
        if not requested or f"{target}@{horizon}" in requested
    )
    unknown = requested - {f"{target}@{horizon}" for target, horizon in RESCUE_CELLS}
    if unknown:
        raise ValueError(f"Unknown failed cells: {sorted(unknown)}")

    reports = {}
    for target, horizon in selected_cells:
        cohort = (
            args.cohort_6h
            if horizon == 6
            else args.cohort_dir / f"mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe.parquet"
        )
        frame = cell_frame(cohort, target, horizon)
        key = f"{target}@{horizon}h"
        print(f"Auditing runtime cell hurdle: {key}", flush=True)
        reports[key] = _audit_horizon(
            frame,
            target=target,
            horizon=horizon,
            patient_seeds=tuple(range(7)),
            hypoxemia_threshold=DEFAULT_HYPOXEMIA_THRESHOLD,
            creatinine_change=TARGET_CHANGE_THRESHOLDS["creatinine"],
            max_rows=int(args.max_rows_per_cell),
        )
        print(json.dumps({key: reports[key]["summary"]}, indent=2), flush=True)

    output = {
        "schema": "mimiciv_failed_cell_hurdle_audit.v1",
        "database": "MIMIC-IV v3.1",
        "cells_evaluated": len(reports),
        "change_thresholds": {
            target: TARGET_CHANGE_THRESHOLDS[target]
            for target, _ in RESCUE_CELLS
        },
        "reports": reports,
        "validated_cells": sorted(
            key for key, report in reports.items() if report["summary"]["validated"]
        ),
        "factual_only": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "validated_cells": output["validated_cells"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
