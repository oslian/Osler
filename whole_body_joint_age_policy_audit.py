"""Audit stale-observation policy for the joint whole-body cohort.

This is a data-contract audit, not a model-promotion test.  It compares the
legacy fixed age cap with a horizon-scaled cap and reports how much usable
state support remains for each target/horizon cell.  The audit never writes
patient-level rows to its JSON output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from whole_body_joint_jepa import load_joint_cohort


DEFAULT_MODULES = (
    "sepsis,aki,respiratory,integumentary_skin_wound,toxic_metabolic,"
    "electrolyte_acid_base,endocrine_stress,gi_pancreatic_nutrition,"
    "cardiac_injury,musculoskeletal_rhabdo,immune_inflammatory,"
    "cardiovascular_instability,acute_neuro,hepatic_failure,coagulopathy_heme"
)


def summarize(frame, variables, horizons):
    output = {}
    for horizon in horizons:
        rows = frame[frame["_source_horizon"] == float(horizon)]
        cells = {}
        for variable in variables:
            current = rows.get(f"{variable}_t")
            future = rows.get(f"future_{variable}")
            if current is None:
                continue
            current_n = int(current.notna().sum())
            future_n = int(future.notna().sum()) if future is not None else 0
            cells[variable] = {
                "rows": int(len(rows)),
                "current_observed": current_n,
                "current_observed_rate": float(current_n / len(rows)) if len(rows) else None,
                "future_observed": future_n,
                "future_observed_rate": float(future_n / len(rows)) if len(rows) else None,
                "median_age_hr": (
                    float(rows[f"{variable}_age_hr"].dropna().median())
                    if f"{variable}_age_hr" in rows and rows[f"{variable}_age_hr"].notna().any()
                    else None
                ),
            }
        output[str(horizon)] = cells
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--modules", default=DEFAULT_MODULES)
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=10000)
    parser.add_argument("--min-modules", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("whole_body_joint_age_policy_audit.json"))
    args = parser.parse_args()
    modules = tuple(item.strip() for item in args.modules.split(",") if item.strip())
    horizons = tuple(int(item.strip()) for item in args.horizons.split(",") if item.strip())
    reports = {}
    for policy in ("fixed", "horizon_scaled"):
        frame, variables, module_names = load_joint_cohort(
            args.data_root,
            modules,
            horizons,
            max_stays_per_horizon=args.max_stays,
            min_modules=args.min_modules,
            alignment="asof",
            asof_max_age_hours=6.0,
            asof_age_policy=policy,
            future_label_tolerance_hours=0.0,
        )
        observed = frame[[f"{v}_t" for v in variables]].notna().to_numpy(dtype=np.float32)
        reports[policy] = {
            "rows": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "modules": list(module_names),
            "variables": list(variables),
            "mean_modules_per_row": float(frame["_module_count"].mean()),
            "current_observed_fraction": float(observed.mean()),
            "by_horizon": summarize(frame, variables, horizons),
        }
    report = {
        "schema": "whole_body_joint_age_policy_audit.v1",
        "policies": {
            "fixed": "all input variables may use up to 6h age",
            "horizon_scaled": "min(6h, max(0.5h, horizon/2)) per variable",
        },
        "reports": reports,
        "stale_observation_guard": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "fixed_rows": reports["fixed"]["rows"],
        "scaled_rows": reports["horizon_scaled"]["rows"],
        "fixed_observed_fraction": reports["fixed"]["current_observed_fraction"],
        "scaled_observed_fraction": reports["horizon_scaled"]["current_observed_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
