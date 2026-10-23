"""Evaluate MIMIC-IV hypoxemia-event cells with the existing factual gate.

MIMIC-IV is single-center, so the available external domains are care unit
and within-stay forward time. The dataset itself is the independent database
stress test for the eICU-derived cell contract. No future treatment fields or
causal claims are permitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from target_specific_domain_conformal_audit import (
    DEFAULT_HYPOXEMIA_THRESHOLD,
    _audit_horizon,
)


def event_frame(cohort: Path, horizon: int) -> pd.DataFrame:
    frame = pd.read_parquet(cohort).copy()
    future_column = f"o2sat_tp{int(horizon)}"
    required = {"subject_id", "stay_id", "t", "o2sat_t", future_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing MIMIC event columns: {missing}")
    frame["anchor_time"] = pd.to_datetime(frame["t"], errors="coerce")
    frame["future_o2sat"] = pd.to_numeric(frame[future_column], errors="coerce")
    frame["horizon_hours"] = int(horizon)
    frame["hospitalid"] = "MIMIC-IV"
    frame["careunit"] = frame.get("first_careunit", "unknown").fillna("unknown")
    return frame[
        frame["anchor_time"].notna()
        & pd.to_numeric(frame["o2sat_t"], errors="coerce").notna()
        & frame["future_o2sat"].notna()
    ].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-1h", type=Path, required=True)
    parser.add_argument("--cohort-3h", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypoxemia-threshold", type=float, default=DEFAULT_HYPOXEMIA_THRESHOLD)
    args = parser.parse_args()

    reports = {}
    for horizon, cohort in ((1, args.cohort_1h), (3, args.cohort_3h)):
        frame = event_frame(cohort, horizon)
        key = f"hypoxemia_event@{horizon}h"
        print(f"Auditing MIMIC event cell: {key}", flush=True)
        reports[key] = _audit_horizon(
            frame,
            target="o2sat",
            horizon=horizon,
            patient_seeds=tuple(range(7)),
            hypoxemia_threshold=float(args.hypoxemia_threshold),
            creatinine_change=0.1,
        )

    output = {
        "schema": "mimiciv_hypoxemia_event_external_audit.v1",
        "database": "MIMIC-IV v3.1",
        "single_center": True,
        "available_external_domains": ["careunit", "forward_time"],
        "unavailable_external_domains": ["hospital"],
        "event_definition": f"future_o2sat <= {float(args.hypoxemia_threshold):g}",
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
