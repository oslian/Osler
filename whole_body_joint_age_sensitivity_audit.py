"""Audit target/horizon-specific as-of age policies without fitting a model.

This is a support and leakage audit. It compares conservative input-age
profiles on the same eICU joint cohort and reports support by target/horizon.
It does not choose a profile from held-out outcomes and never writes patient
rows. A profile is only eligible for later model promotion after an
independent forecast gate is rerun with that profile.
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


def _variable_limits(variables: tuple[str, ...], profile: str, max_age: float) -> dict[str, float] | None:
    if profile == "horizon_scaled":
        return None
    if profile.startswith("global_"):
        cap = float(profile.split("_", 1)[1])
        return {variable: cap for variable in variables}
    if profile == "target_specific_conservative":
        slow_tokens = (
            "creatinine", "bun", "urine_output", "bilirubin", "troponin",
            "ck_mb", "inr", "ptt", "fibrinogen", "ferritin", "crp",
            "albumin", "total_protein", "tsh", "t4", "cortisol", "pth",
            "hgb", "hemoglobin", "hematocrit", "platelets",
        )
        return {
            variable: min(float(max_age), 12.0 if any(token in variable.lower() for token in slow_tokens) else 3.0)
            for variable in variables
        }
    raise ValueError(f"unknown profile: {profile}")


def _summary(frame, variables, horizons):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--modules", default=DEFAULT_MODULES)
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=10000)
    parser.add_argument("--min-modules", type=int, default=2)
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument(
        "--profiles",
        default="global_0.5,global_1,global_2,global_3,global_6,global_12,horizon_scaled,target_specific_conservative",
    )
    parser.add_argument("--output", type=Path, default=Path("whole_body_joint_age_sensitivity_audit.json"))
    args = parser.parse_args()
    modules = tuple(item.strip() for item in args.modules.split(",") if item.strip())
    horizons = tuple(int(item.strip()) for item in args.horizons.split(",") if item.strip())
    profiles = tuple(item.strip() for item in args.profiles.split(",") if item.strip())

    reports = {}
    variables = None
    for profile in profiles:
        if variables is None:
            _, variables, _ = load_joint_cohort(
                args.data_root,
                modules,
                horizons,
                max_stays_per_horizon=args.max_stays,
                min_modules=args.min_modules,
                alignment="asof",
                asof_max_age_hours=float(args.max_age_hours),
                asof_age_policy="horizon_scaled",
                future_label_tolerance_hours=0.0,
            )
        frame, loaded_variables, module_names = load_joint_cohort(
            args.data_root,
            modules,
            horizons,
            max_stays_per_horizon=args.max_stays,
            min_modules=args.min_modules,
            alignment="asof",
            asof_max_age_hours=float(args.max_age_hours),
            asof_age_policy="horizon_scaled" if profile == "horizon_scaled" else "fixed",
            variable_age_limits=(
                _variable_limits(tuple(variables), profile, args.max_age_hours)
                if profile != "horizon_scaled" else None
            ),
            future_label_tolerance_hours=0.0,
        )
        variables = tuple(loaded_variables)
        observed = frame[[f"{v}_t" for v in variables]].notna().to_numpy(dtype=np.float32)
        reports[profile] = {
            "rows": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "modules": list(module_names),
            "variables": list(variables),
            "mean_modules_per_row": float(frame["_module_count"].mean()),
            "current_observed_fraction": float(observed.mean()) if observed.size else None,
            "by_horizon": _summary(frame, variables, horizons),
        }

    report = {
        "schema": "whole_body_joint_age_sensitivity_audit.v1",
        "profiles": list(profiles),
        "reports": reports,
        "future_label_tolerance_hours": 0.0,
        "future_label_leakage_guard": True,
        "selection_policy": "support_only; no outcome-based profile promotion",
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "profiles": {
            name: {
                "rows": value["rows"],
                "observed_fraction": value["current_observed_fraction"],
            }
            for name, value in reports.items()
        },
        "causal_claim_allowed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
