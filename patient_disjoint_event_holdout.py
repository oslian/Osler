"""Build a patient-disjoint holdout matching the joint loader stay sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_patient_disjoint_holdout(
    frame: pd.DataFrame,
    *,
    sampled_stays: int,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if frame.empty:
        raise ValueError("event examples are empty")
    stay_column = "stay_id" if "stay_id" in frame else "encounter_id"
    patient_column = "subject_id" if "subject_id" in frame else "patient_id"
    stays = frame[stay_column].astype(str).drop_duplicates().to_numpy()
    if sampled_stays <= 0 or sampled_stays >= len(stays):
        raise ValueError("sampled_stays must leave a non-empty holdout")
    selected = set(
        np.random.default_rng(seed).choice(
            stays, size=int(sampled_stays), replace=False
        )
    )
    selected_patients = set(
        frame.loc[
            frame[stay_column].astype(str).isin(selected), patient_column
        ].astype(str)
    )
    holdout = frame[
        ~frame[patient_column].astype(str).isin(selected_patients)
    ].copy()
    if holdout.empty:
        raise ValueError("patient-disjoint holdout is empty")
    report = {
        "source_rows": int(len(frame)),
        "source_stays": int(frame[stay_column].astype(str).nunique()),
        "source_patients": int(frame[patient_column].astype(str).nunique()),
        "development_stays_sampled": int(len(selected)),
        "development_patients_excluded": int(len(selected_patients)),
        "holdout_rows": int(len(holdout)),
        "holdout_stays": int(holdout[stay_column].astype(str).nunique()),
        "holdout_patients": int(holdout[patient_column].astype(str).nunique()),
        "patient_overlap": 0,
        "row_level_identifiers_in_report": False,
    }
    return holdout.reset_index(drop=True), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sampled-stays", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    holdout, report = build_patient_disjoint_holdout(
        frame, sampled_stays=args.sampled_stays, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_parquet(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
