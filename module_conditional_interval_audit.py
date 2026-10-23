"""Disease-module-conditional intervals (v3) for frozen MIMIC recipes.

v2 intervals scale MIMIC-only conformal residuals by current-value bin.  eICU
under-coverage that remains is concentrated in each target's own disease
module (creatinine in AKI, glucose in endocrine stress, ...).  v3 calibrates
separately for patients who are and are not members of the target's disease
module (``PREFERRED_MODULE``), each with the v2 value normalization:

- member calibration pool: MIMIC calibration rows of member stays plus up to
  15,000 rows of member stays from MIMIC patients never used for fitting;
- non-member pool: MIMIC calibration rows of non-member stays;
- a group with fewer than 100 calibration patients falls back to v2.

Module membership comes from ``icu_disease_module_membership.py`` (MIMIC,
NWICU) or from the eICU module cohort files (eICU).  The MIMIC member pool also
gives a MIMIC-only held-out check of v2 on disease-module patients.

eICU and NWICU were both used before, so external results here are
non-independent re-checks and cannot support promotion.  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from osler_jepa.recipe_availability import deployable_recipe

from external_value_conditional_interval_audit import (
    GATE_AXES,
    EXTERNAL_PREFIX,
    external_cohort,
    evaluability,
    interval_report,
    normalized_bin_widths,
    value_bin_edges,
)
from mimic_to_eicu_49_cell_external_audit import (
    PREFERRED_MODULE,
    cap_by_subject,
    fit_continuous_recipe,
    group_equal_coverage,
    normalize_cohort,
    stable_seed,
)
from target_specific_domain_conformal_audit import _safe_float


MIN_GROUP_PATIENTS = 100
EXTRA_MEMBER_ROWS = 15000
CV_FOLDS = 5


def raw_stay(frame: pd.DataFrame) -> np.ndarray:
    text = frame["stay_id"].astype(str).str.split(":", n=1).str[-1]
    return pd.to_numeric(text, errors="coerce").to_numpy()


def load_members(args: argparse.Namespace, module: str) -> dict[str, set[int]]:
    mimic = pd.read_parquet(args.mimic_membership)
    members = {"MIMIC-IV": set(mimic.loc[mimic["module"] == module, "stay_id"].astype(int))}
    if args.database == "NWICU":
        nwicu = pd.read_parquet(args.nwicu_membership)
        members["NWICU"] = set(nwicu.loc[nwicu["module"] == module, "stay_id"].astype(int))
    else:
        stays: set[int] = set()
        for path in sorted(args.eicu_root.glob(f"eicu_{module}_transitions_*h.parquet")):
            stays |= set(pd.read_parquet(path, columns=["stay_id"])["stay_id"].astype(int))
        members["eICU"] = stays
    return members


def group_widths(
    frame: pd.DataFrame,
    pool_rows: np.ndarray,
    pool_residual: np.ndarray,
    query_rows: np.ndarray,
    target: str,
) -> np.ndarray:
    current = _safe_float(frame, f"{target}_t")
    edges = value_bin_edges(current[pool_rows])
    widths, _ = normalized_bin_widths(
        frame, pool_rows, pool_residual,
        np.digitize(current[pool_rows], edges), np.digitize(current[query_rows], edges),
    )
    return widths


def cross_validated_member_coverage(frame, rows, residual, target, seed) -> float:
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    np.random.default_rng(seed).shuffle(unique)
    covered = np.zeros(len(rows), dtype=bool)
    for heldout in np.array_split(unique, CV_FOLDS):
        test = np.isin(subjects, heldout)
        covered[test] = residual[test] <= group_widths(frame, rows[~test], residual[~test], rows[test], target)
    return group_equal_coverage(frame, rows, covered, "subject_id")


def evaluate_cell(
    raw_mimic: pd.DataFrame,
    mimic: pd.DataFrame,
    external: pd.DataFrame,
    members: dict[str, set[int]],
    *,
    cell: str,
    target: str,
    horizon: int,
    source: str,
    database: str,
) -> dict:
    used_subjects = set(mimic["subject_id"].astype(str))
    pool, _ = normalize_cohort(raw_mimic, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
    pool = pool[~pool["subject_id"].astype(str).isin(used_subjects)]
    pool = pool[np.isin(raw_stay(pool), list(members["MIMIC-IV"]))]
    pool = cap_by_subject(pool, EXTRA_MEMBER_ROWS, stable_seed(cell, "member_pool"))
    pool = pool.assign(subject_id="MIMICPOOL:" + pool["subject_id"].astype(str).str.split(":", n=1).str[-1])

    frame = pd.concat([mimic, pool, external], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    subject = frame["subject_id"].astype(str)
    train = np.flatnonzero(subject.str.startswith("MIMIC:"))
    pool_rows_all = np.flatnonzero(subject.str.startswith("MIMICPOOL:"))
    external_rows_all = np.flatnonzero(subject.str.startswith(EXTERNAL_PREFIX[database]))
    fitted = fit_continuous_recipe(
        frame, train, np.concatenate([pool_rows_all, external_rows_all]),
        target=target, source=source, seed=stable_seed(cell, "mimic_to_eicu"),
    )
    future = _safe_float(frame, f"future_{target}")
    prediction = pd.Series(fitted["test_prediction"], index=fitted["test_rows"])
    residual_of = lambda rows: np.abs(prediction[rows].to_numpy() - future[rows])
    pool_rows = np.intersect1d(fitted["test_rows"], pool_rows_all)
    external_rows = np.intersect1d(fitted["test_rows"], external_rows_all)

    cal = fitted["calibration_rows"]
    cal_residual = np.abs(fitted["calibration_prediction"] - future[cal])
    cal_member = np.isin(raw_stay(frame.iloc[cal]), list(members["MIMIC-IV"]))
    external_member = np.isin(raw_stay(frame.iloc[external_rows]), list(members[database]))
    external_residual = residual_of(external_rows)

    v2 = group_widths(frame, cal, cal_residual, external_rows, target)
    member_pool_rows = np.concatenate([cal[cal_member], pool_rows])
    member_pool_residual = np.concatenate([cal_residual[cal_member], residual_of(pool_rows)])
    groups = {
        "member": (member_pool_rows, member_pool_residual, external_member),
        "non_member": (cal[~cal_member], cal_residual[~cal_member], ~external_member),
    }
    v3 = v2.copy()
    fallback = {}
    for name, (rows, residual, mask) in groups.items():
        patients = frame.iloc[rows]["subject_id"].nunique() if len(rows) else 0
        fallback[name] = bool(patients < MIN_GROUP_PATIENTS)
        if mask.any() and not fallback[name]:
            v3[mask] = group_widths(frame, rows, residual, external_rows[mask], target)

    axes = GATE_AXES[database]["interval"]
    report = {
        "external_rows": int(len(external_rows)),
        "external_member_share": float(np.mean(external_member)),
        "mimic_member_pool_patients": int(frame.iloc[member_pool_rows]["subject_id"].nunique()),
        "mimic_non_member_calibration_patients": int(frame.iloc[cal[~cal_member]]["subject_id"].nunique()),
        "group_fallback_to_v2": fallback,
        "v2": interval_report(frame, external_rows, external_residual <= v2, v2, axes),
        "v3": interval_report(frame, external_rows, external_residual <= v3, v3, axes),
    }
    if len(pool_rows):
        pool_v2 = group_widths(frame, cal, cal_residual, pool_rows, target)
        report["mimic_heldout_members_v2_coverage"] = group_equal_coverage(
            frame, pool_rows, residual_of(pool_rows) <= pool_v2, "subject_id"
        )
        report["mimic_heldout_members_patients"] = int(frame.iloc[pool_rows]["subject_id"].nunique())
    if not fallback["member"]:
        report["mimic_member_v3_cross_validated_coverage"] = cross_validated_member_coverage(
            frame, member_pool_rows, member_pool_residual, target, stable_seed(cell, "member_cv")
        )
    for name in ("member", "non_member"):
        mask = external_member if name == "member" else ~external_member
        if mask.sum() >= 30:
            report[f"external_{name}_coverage_v2"] = group_equal_coverage(frame, external_rows[mask], (external_residual <= v2)[mask], "subject_id")
            report[f"external_{name}_coverage_v3"] = group_equal_coverage(frame, external_rows[mask], (external_residual <= v3)[mask], "subject_id")
    return report


def write_markdown(report: dict, path: Path) -> None:
    database = report["external_database"]
    lines = [
        f"# Module-Conditional Intervals (v3) on {database} — non-independent re-check",
        "",
        f"- Evaluable continuous cells: `{report['evaluable_cells']}`",
        f"- Interval pass v2 → v3: `{report['v2_interval_pass_cells']}` → `{report['v3_interval_pass_cells']}`",
        f"- Full pass (point gate from the v2 run) v2 → v3: `{report['v2_full_pass_cells']}` → `{report['v3_full_pass_cells']}`",
        f"- v2 reproduces the v2 audit in every cell: `{report['v2_reproduces_v2_audit']}`",
        "",
        "| Cell | Module | Members | MIMIC held-out members, v2 cov. | MIMIC members, v3 CV | Ext. patient cov. v2 → v3 | Ext. group cov. v2 → v3 | Full v2 → v3 |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    group_key = "hospital_equal_coverage" if database == "eICU" else "careunit_equal_coverage"
    fmt = lambda value: "-" if value is None else f"{value:.3f}"
    for cell in report["cells"]:
        if not cell["evaluable"]:
            continue
        m = cell["metrics"]
        lines.append(
            f"| {cell['cell']} | {cell['module']} | {m['external_member_share']:.0%} | "
            f"{fmt(m.get('mimic_heldout_members_v2_coverage'))} | {fmt(m.get('mimic_member_v3_cross_validated_coverage'))} | "
            f"{m['v2']['patient_equal_coverage']:.3f} → {m['v3']['patient_equal_coverage']:.3f} | "
            f"{m['v2'][group_key]:.3f} → {m['v3'][group_key]:.3f} | {cell['v2_full_pass']} → {cell['v3_full_pass']} |"
        )
    lines += ["", "Non-independent: this database was already used. No promotion. `causal_claim_allowed: false`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    v2_audit = {cell["cell"]: cell for cell in json.loads(args.v2_audit.read_text(encoding="utf-8"))["cells"]}
    contracts = [row for row in development["cells"] if row["task_type"] != "binary_event_probability"]
    by_horizon: dict[int, list[dict]] = {}
    for row in contracts:
        by_horizon.setdefault(int(row["horizon_hours"]), []).append(row)
    membership_cache: dict[str, dict[str, set[int]]] = {}
    cells = []
    for horizon in sorted(by_horizon):
        raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        for contract in sorted(by_horizon[horizon], key=lambda row: row["target"]):
            cell, target, source = contract["cell"], contract["target"], contract["source_model"]
            module = PREFERRED_MODULE.get(target, "sepsis")
            base = {"cell": cell, "module": module}
            mimic, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=15000)
            external, _ = external_cohort(args, target, horizon)
            source, _ = deployable_recipe(source, external, target)
            reason = evaluability(args.database, external)
            if reason:
                cells.append({**base, "evaluable": False, "reason": reason})
                continue
            print(f"{args.database} v3 {cell} ({module})", flush=True)
            if module not in membership_cache:
                membership_cache[module] = load_members(args, module)
            metrics = evaluate_cell(
                raw, mimic, external, membership_cache[module],
                cell=cell, target=target, horizon=horizon, source=source, database=args.database,
            )
            frozen_v2 = v2_audit[cell]["metrics"]["value_conditional"]
            reproduces = all(
                abs(metrics["v2"][key] - frozen_v2[key]) < 1e-9
                for key in ("patient_equal_coverage", "hospital_equal_coverage", "careunit_equal_coverage")
            )
            point = bool(v2_audit[cell]["metrics"]["point_pass"])
            cells.append({
                **base,
                "evaluable": True,
                "metrics": metrics,
                "point_pass_from_v2_run": point,
                "v2_reproduces_v2_audit": reproduces,
                "v2_full_pass": bool(point and metrics["v2"]["pass"]),
                "v3_full_pass": bool(point and metrics["v3"]["pass"]),
            })
        del raw
    evaluable = [cell for cell in cells if cell["evaluable"]]
    output = {
        "schema": "module_conditional_interval_audit.v1",
        "external_database": args.database,
        "interval_method": "module_membership_x_normalized_bin_scale.v3",
        "independent_validation": False,
        "evaluable_cells": len(evaluable),
        "not_evaluable_reasons": dict(Counter(cell["reason"] for cell in cells if not cell["evaluable"])),
        "v2_reproduces_v2_audit": all(cell["v2_reproduces_v2_audit"] for cell in evaluable),
        "v2_interval_pass_cells": sum(cell["metrics"]["v2"]["pass"] for cell in evaluable),
        "v3_interval_pass_cells": sum(cell["metrics"]["v3"]["pass"] for cell in evaluable),
        "v2_full_pass_cells": sum(cell["v2_full_pass"] for cell in evaluable),
        "v3_full_pass_cells": sum(cell["v3_full_pass"] for cell in evaluable),
        "cells": cells,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
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
    parser.add_argument("--mimic-membership", type=Path, default=Path("mimiciv_disease_module_membership.parquet"))
    parser.add_argument("--nwicu-membership", type=Path, default=Path("nwicu_disease_module_membership.parquet"))
    parser.add_argument("--v2-audit", type=Path, required=True)
    parser.add_argument("--max-rows-per-database", type=int, default=15000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
