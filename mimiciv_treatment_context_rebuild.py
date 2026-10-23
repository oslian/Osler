"""Rebuild unit-safe pre-anchor MIMIC-IV treatment context in place.

The observation state and future labels are reused from an existing cohort;
only ``hist_*``/``act_*`` columns are replaced.  This avoids rescanning the
very large measurement tables when treatment extraction changes.  Output is
still a factual observational cohort and grants no causal authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mimiciv_observation_transition_extract import (
    MIMIC_DIR,
    assemble_treatment_context,
    con,
    pull_treatment_evidence,
    register_stays,
)
from osler_jepa.treatment_units import TARGET_TREATMENT_ACTIONS


FORECAST_ACTIONS = set().union(*TARGET_TREATMENT_ACTIONS.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=MIMIC_DIR)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--horizon-hours", type=float, default=6.0)
    return parser.parse_args()


def _coverage(evidence: pd.DataFrame) -> dict[str, object]:
    by_action: dict[str, object] = {}
    for action, group in evidence.groupby("action", sort=True):
        by_action[str(action)] = {
            "rows": int(len(group)),
            "stays": int(group["stay_id"].nunique()),
            "canonical_dose_rows": int(group["canonical_dose_value"].notna().sum()),
            "canonical_rate_rows": int(group["canonical_rate_value"].notna().sum()),
            "dose_dimensions": {
                str(key): int(value)
                for key, value in group["canonical_dose_dimension"]
                .dropna()
                .value_counts()
                .sort_index()
                .items()
            },
            "rate_dimensions": {
                str(key): int(value)
                for key, value in group["canonical_rate_dimension"]
                .dropna()
                .value_counts()
                .sort_index()
                .items()
            },
        }
    return by_action


def main() -> None:
    args = parse_args()
    if args.horizon_hours <= 0.0:
        raise ValueError("--horizon-hours must be positive")

    anchors = pd.read_parquet(args.base, columns=["subject_id", "stay_id", "t"])
    anchors["t"] = pd.to_datetime(anchors["t"])
    connection = con()
    register_stays(connection, anchors[["stay_id"]].drop_duplicates())
    evidence = pull_treatment_evidence(
        connection,
        args.mimic_dir,
        actions=set(FORECAST_ACTIONS),
    )
    print(
        f"Treatment evidence: {len(evidence):,} rows; "
        f"{evidence['stay_id'].nunique():,} stays"
    )
    context = assemble_treatment_context(
        anchors,
        evidence,
        horizon_hours=float(args.horizon_hours),
    )
    print(f"Treatment context: {len(context):,} rows; {len(context.columns):,} columns")

    base = pd.read_parquet(args.base)
    obsolete = [
        column
        for column in base.columns
        if column.startswith("hist_") or column.startswith("act_")
    ]
    base = base.drop(columns=obsolete)
    output = pd.concat(
        [base.reset_index(drop=True), context.reset_index(drop=True)], axis=1
    )
    output.to_parquet(args.output, index=False)

    report = {
        "artifact": "MIMIC-IV unit-safe factual treatment context rebuild",
        "base": str(args.base.resolve()),
        "output": str(args.output.resolve()),
        "mimic_dir": str(args.mimic_dir.resolve()),
        "rows": int(len(output)),
        "subjects": int(output["subject_id"].nunique()),
        "stays": int(output["stay_id"].nunique()),
        "removed_obsolete_treatment_columns": int(len(obsolete)),
        "new_treatment_columns": int(len(context.columns)),
        "forecast_actions": sorted(FORECAST_ACTIONS),
        "evidence_rows": int(len(evidence)),
        "evidence_stays": int(evidence["stay_id"].nunique()),
        "source_rows": {
            str(key): int(value)
            for key, value in evidence["source"].value_counts().sort_index().items()
        },
        "action_coverage": _coverage(evidence),
        "timing_contract": {
            "history_only_for_forecasting": True,
            "future_treatment_features_are_labels_not_inputs": True,
            "open_inputevent_final_amount_excluded": True,
            "open_inputevent_pre_anchor_exposure_from_observed_rate": True,
        },
        "mixed_unit_numeric_sums_removed": True,
        "causal_claim_allowed": False,
        "row_level_outputs_committed": False,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()

