"""As-of disease-activity-conditional intervals (v4) for frozen MIMIC recipes.

v3 conditioned on diagnosis-based module membership, which is not portable
across databases and uses discharge codes unavailable at prediction time.  v4
conditions on the eICU module "active" rules evaluated on as-of measurements
(``*_t`` values available before the anchor), keeping only measurement criteria
so the rule is identical in MIMIC-IV, eICU and NWICU:

- AKI: creatinine >= 1.5, urine output < 30, potassium >= 5.5, BUN >= 40;
- respiratory: SpO2 <= 92, respiratory rate >= 24, pH <= 7.30, PaCO2 >= 50,
  FiO2 >= 40, PEEP >= 5;
- other modules: ``active_low`` / ``active_high`` of BODY_SYSTEM_CONFIGS.

Treatment and ventilator-mode criteria are dropped (absent in NWICU); a missing
measurement never makes a row active.  Each target uses its own module
(``PREFERRED_MODULE``).  Active and inactive rows get separate v2-style
(normalized value-bin) conformal widths, calibrated on MIMIC only.

Calibration: MIMIC calibration rows plus half of up to 15,000 rows from MIMIC
patients never used for fitting; the other half is a MIMIC held-out check.

Predeclared adoption rule (written before any v4 result was seen):
1. on held-out MIMIC patients, active- and inactive-group coverage both lie in
   [0.89, 0.91] in at least 90% of cells (groups with >= 100 patients);
2. full passes on the (non-independent) eICU and NWICU re-checks are not lower
   than v2's.

Factual research-only; no promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from osler_jepa.recipe_availability import deployable_recipe

from eicu_body_system_configs import BODY_SYSTEM_CONFIGS
from external_value_conditional_interval_audit import (
    EXTERNAL_PREFIX,
    GATE_AXES,
    evaluability,
    external_cohort,
    interval_report,
)
from mimic_to_eicu_49_cell_external_audit import (
    PREFERRED_MODULE,
    cap_by_subject,
    fit_continuous_recipe,
    group_equal_coverage,
    normalize_cohort,
    stable_seed,
)
from module_conditional_interval_audit import group_widths
from target_specific_domain_conformal_audit import _safe_float


MIN_GROUP_PATIENTS = 100
EXTRA_ROWS = 15000
CRITERIA: dict[str, tuple[tuple[str, str, float], ...]] = {
    "aki": (
        ("creatinine", ">=", 1.5), ("urine_output", "<", 30.0),
        ("potassium", ">=", 5.5), ("bun", ">=", 40.0),
    ),
    "respiratory": (
        ("o2sat", "<=", 92.0), ("respiratory_rate", ">=", 24.0), ("ph", "<=", 7.30),
        ("paco2", ">=", 50.0), ("fio2", ">=", 40.0), ("peep", ">=", 5.0),
    ),
    **{
        name: tuple(
            [(var, "<=", value) for var, value in BODY_SYSTEM_CONFIGS[name].active_low.items()]
            + [(var, ">=", value) for var, value in BODY_SYSTEM_CONFIGS[name].active_high.items()]
        )
        for name in ("endocrine_stress", "cardiovascular_instability", "coagulopathy_heme", "electrolyte_acid_base")
    },
}
OPERATORS = {">=": np.greater_equal, "<=": np.less_equal, "<": np.less}


def active_rows(frame: pd.DataFrame, module: str) -> np.ndarray:
    """As-of module activity from measurements only; missing values never activate."""

    active = np.zeros(len(frame), dtype=bool)
    for var, operator, threshold in CRITERIA[module]:
        values = _safe_float(frame, f"{var}_t")
        with np.errstate(invalid="ignore"):
            active |= np.isfinite(values) & OPERATORS[operator](values, threshold)
    return active


def evaluate_cell(raw_mimic, mimic, external, *, cell, target, horizon, source, database) -> dict:
    module = PREFERRED_MODULE.get(target, "sepsis")
    used = set(mimic["subject_id"].astype(str))
    extra, _ = normalize_cohort(raw_mimic, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
    extra = cap_by_subject(extra[~extra["subject_id"].astype(str).isin(used)], EXTRA_ROWS, stable_seed(cell, "v4_extra"))
    subjects = np.unique(extra["subject_id"].astype(str))
    rng = np.random.default_rng(stable_seed(cell, "v4_extra_split"))
    holdout = set(rng.choice(subjects, size=len(subjects) // 2, replace=False))
    extra = extra.assign(
        subject_id=np.where(extra["subject_id"].astype(str).isin(holdout), "MIMICHOLD:", "MIMICPOOL:")
        + extra["subject_id"].astype(str).str.split(":", n=1).str[-1]
    )
    frame = pd.concat([mimic, extra, external], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    subject = frame["subject_id"].astype(str)
    train = np.flatnonzero(subject.str.startswith("MIMIC:"))
    pool_all = np.flatnonzero(subject.str.startswith("MIMICPOOL:"))
    hold_all = np.flatnonzero(subject.str.startswith("MIMICHOLD:"))
    ext_all = np.flatnonzero(subject.str.startswith(EXTERNAL_PREFIX[database]))
    fitted = fit_continuous_recipe(
        frame, train, np.concatenate([pool_all, hold_all, ext_all]),
        target=target, source=source, seed=stable_seed(cell, "mimic_to_eicu"),
    )
    future = _safe_float(frame, f"future_{target}")
    prediction = pd.Series(fitted["test_prediction"], index=fitted["test_rows"])
    residual_of = lambda rows: np.abs(prediction[rows].to_numpy() - future[rows])
    pool = np.intersect1d(fitted["test_rows"], pool_all)
    hold = np.intersect1d(fitted["test_rows"], hold_all)
    ext = np.intersect1d(fitted["test_rows"], ext_all)
    cal = fitted["calibration_rows"]
    cal_residual = np.abs(fitted["calibration_prediction"] - future[cal])

    calibration_rows = np.concatenate([cal, pool])
    calibration_residual = np.concatenate([cal_residual, residual_of(pool)])
    calibration_active = active_rows(frame.iloc[calibration_rows], module)

    def widths(query: np.ndarray, conditional: bool) -> tuple[np.ndarray, dict]:
        if not conditional:
            return group_widths(frame, cal, cal_residual, query, target), {}
        query_active = active_rows(frame.iloc[query], module)
        output = group_widths(frame, cal, cal_residual, query, target)
        fallback = {}
        for name, flag in (("active", True), ("inactive", False)):
            pool_mask = calibration_active == flag
            patients = frame.iloc[calibration_rows[pool_mask]]["subject_id"].nunique()
            fallback[name] = bool(patients < MIN_GROUP_PATIENTS)
            mask = query_active == flag
            if mask.any() and not fallback[name]:
                output[mask] = group_widths(
                    frame, calibration_rows[pool_mask], calibration_residual[pool_mask], query[mask], target
                )
        return output, fallback

    def group_coverage(rows: np.ndarray, covered: np.ndarray) -> dict:
        active = active_rows(frame.iloc[rows], module)
        result = {"active_share": float(np.mean(active))}
        for name, mask in (("active", active), ("inactive", ~active)):
            patients = frame.iloc[rows[mask]]["subject_id"].nunique() if mask.any() else 0
            result[f"{name}_patients"] = int(patients)
            result[f"{name}_coverage"] = (
                group_equal_coverage(frame, rows[mask], covered[mask], "subject_id")
                if patients >= MIN_GROUP_PATIENTS else None
            )
        return result

    hold_residual, ext_residual = residual_of(hold), residual_of(ext)
    hold_v2, _ = widths(hold, False)
    hold_v4, _ = widths(hold, True)
    ext_v2, _ = widths(ext, False)
    ext_v4, fallback = widths(ext, True)
    axes = GATE_AXES[database]["interval"]
    return {
        "module": module,
        "group_fallback_to_v2": fallback,
        "mimic_holdout": {
            "v2": {**group_coverage(hold, hold_residual <= hold_v2),
                   "patient_equal_coverage": group_equal_coverage(frame, hold, hold_residual <= hold_v2, "subject_id")},
            "v4": {**group_coverage(hold, hold_residual <= hold_v4),
                   "patient_equal_coverage": group_equal_coverage(frame, hold, hold_residual <= hold_v4, "subject_id")},
        },
        "external": {
            "v2": {**interval_report(frame, ext, ext_residual <= ext_v2, ext_v2, axes), **group_coverage(ext, ext_residual <= ext_v2)},
            "v4": {**interval_report(frame, ext, ext_residual <= ext_v4, ext_v4, axes), **group_coverage(ext, ext_residual <= ext_v4)},
        },
    }


def rule_one_passes(cell: dict) -> bool | None:
    groups = [cell["metrics"]["mimic_holdout"]["v4"].get(f"{name}_coverage") for name in ("active", "inactive")]
    groups = [value for value in groups if value is not None]
    return all(0.89 <= value <= 0.91 for value in groups) if groups else None


def write_markdown(report: dict, path: Path) -> None:
    database = report["external_database"]
    fmt = lambda value: "-" if value is None else f"{value:.3f}"
    lines = [
        f"# Activity-Conditional Intervals (v4): MIMIC held-out check and {database} re-check",
        "",
        f"- Cells: `{report['evaluable_cells']}`",
        f"- Rule 1 (MIMIC held-out active and inactive coverage in [0.89, 0.91]): "
        f"`{report['rule_one_cells_passing']}` of `{report['rule_one_cells_assessed']}` cells "
        f"→ {'met' if report['rule_one_met'] else 'NOT met'} (needs ≥ 90%)",
        f"- {database} full pass v2 → v4: `{report['v2_full_pass_cells']}` → `{report['v4_full_pass_cells']}` "
        f"(non-independent) → rule 2 {'met' if report['rule_two_met'] else 'NOT met'}",
        f"- v2 reproduces the v2 audit in every cell: `{report['v2_reproduces_v2_audit']}`",
        "",
        "| Cell | Module | MIMIC active share | MIMIC hold-out active v2 → v4 | inactive v2 → v4 | Ext. active share | Ext. patient cov. v2 → v4 | Full v2 → v4 |",
        "|---|---|---:|---|---|---:|---|---|",
    ]
    for cell in report["cells"]:
        if not cell["evaluable"]:
            continue
        hold, ext = cell["metrics"]["mimic_holdout"], cell["metrics"]["external"]
        lines.append(
            f"| {cell['cell']} | {cell['metrics']['module']} | {hold['v2']['active_share']:.0%} | "
            f"{fmt(hold['v2']['active_coverage'])} → {fmt(hold['v4']['active_coverage'])} | "
            f"{fmt(hold['v2']['inactive_coverage'])} → {fmt(hold['v4']['inactive_coverage'])} | "
            f"{ext['v2']['active_share']:.0%} | {ext['v2']['patient_equal_coverage']:.3f} → {ext['v4']['patient_equal_coverage']:.3f} | "
            f"{cell['v2_full_pass']} → {cell['v4_full_pass']} |"
        )
    lines += ["", "External database already used: non-independent. No promotion. `causal_claim_allowed: false`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    v2_audit = {cell["cell"]: cell for cell in json.loads(args.v2_audit.read_text(encoding="utf-8"))["cells"]}
    contracts = [row for row in development["cells"] if row["task_type"] != "binary_event_probability"]
    by_horizon: dict[int, list[dict]] = {}
    for row in contracts:
        by_horizon.setdefault(int(row["horizon_hours"]), []).append(row)
    cells = []
    for horizon in sorted(by_horizon):
        raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        for contract in sorted(by_horizon[horizon], key=lambda row: row["target"]):
            cell, target, source = contract["cell"], contract["target"], contract["source_model"]
            mimic, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=15000)
            external, _ = external_cohort(args, target, horizon)
            source, _ = deployable_recipe(source, external, target)
            reason = evaluability(args.database, external)
            if reason:
                cells.append({"cell": cell, "evaluable": False, "reason": reason})
                continue
            print(f"{args.database} v4 {cell}", flush=True)
            metrics = evaluate_cell(
                raw, mimic, external, cell=cell, target=target, horizon=horizon, source=source, database=args.database
            )
            frozen = v2_audit[cell]["metrics"]["value_conditional"]
            reproduces = all(
                abs(metrics["external"]["v2"][key] - frozen[key]) < 1e-9
                for key in ("patient_equal_coverage", "hospital_equal_coverage", "careunit_equal_coverage")
            )
            point = bool(v2_audit[cell]["metrics"]["point_pass"])
            cells.append({
                "cell": cell,
                "evaluable": True,
                "metrics": metrics,
                "v2_reproduces_v2_audit": reproduces,
                "v2_full_pass": bool(point and metrics["external"]["v2"]["pass"]),
                "v4_full_pass": bool(point and metrics["external"]["v4"]["pass"]),
            })
        del raw
    evaluable = [cell for cell in cells if cell["evaluable"]]
    assessed = [rule_one_passes(cell) for cell in evaluable]
    assessed = [value for value in assessed if value is not None]
    output = {
        "schema": "activity_conditional_interval_audit.v1",
        "external_database": args.database,
        "interval_method": "asof_measurement_activity_x_normalized_bin_scale.v4",
        "independent_validation": False,
        "evaluable_cells": len(evaluable),
        "v2_reproduces_v2_audit": all(cell["v2_reproduces_v2_audit"] for cell in evaluable),
        "rule_one_cells_assessed": len(assessed),
        "rule_one_cells_passing": int(sum(assessed)),
        "rule_one_met": bool(assessed and sum(assessed) >= 0.9 * len(assessed)),
        "v2_full_pass_cells": sum(cell["v2_full_pass"] for cell in evaluable),
        "v4_full_pass_cells": sum(cell["v4_full_pass"] for cell in evaluable),
        "cells": cells,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    output["rule_two_met"] = output["v4_full_pass_cells"] >= output["v2_full_pass_cells"]
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_markdown(output, args.findings)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", choices=sorted(GATE_AXES), required=True)
    parser.add_argument(
        "--development-report",
        type=Path,
        default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"),
    )
    parser.add_argument(
        "--mimic-template",
        default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v2.parquet",
    )
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_future_label_repaired"))
    parser.add_argument("--external-template", default=None, help="parquet path template with {horizon} for single-site databases")
    parser.add_argument("--v2-audit", type=Path, required=True)
    parser.add_argument("--max-rows-per-database", type=int, default=15000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
