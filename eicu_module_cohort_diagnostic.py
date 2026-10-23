"""Same MIMIC recipe, same target, different eICU disease-module cohorts.

eICU transition files are built per disease module (AKI, sepsis, ...), unlike
whole-ICU MIMIC and NWICU.  If v2 coverage for one target differs strongly
between modules, cohort definition drives part of the eICU gap.

eICU labels are used for analysis only.  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_within_bin_gap_diagnostic import v2_widths
from mimic_to_eicu_49_cell_external_audit import (
    MIN_ROWS,
    MIN_SUBJECTS,
    eicu_path,
    fit_continuous_recipe,
    group_equal_coverage,
    normalize_cohort,
    stable_seed,
)
from target_specific_domain_conformal_audit import _safe_float


MODULES = (
    "aki", "sepsis", "cardiovascular_instability", "respiratory", "endocrine_stress",
    "electrolyte_acid_base", "coagulopathy_heme", "hepatic_failure", "acute_neuro",
)


def score(mimic: pd.DataFrame, eicu: pd.DataFrame, *, cell: str, target: str, source: str) -> dict:
    frame = pd.concat([mimic, eicu], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    train = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("MIMIC:"))
    test = np.flatnonzero(frame["subject_id"].astype(str).str.startswith("EICU:"))
    fitted = fit_continuous_recipe(frame, train, test, target=target, source=source, seed=stable_seed(cell, "mimic_to_eicu"))
    rows = fitted["test_rows"]
    future = _safe_float(frame, f"future_{target}")[rows]
    current = _safe_float(frame, f"{target}_t")[rows]
    covered = np.abs(fitted["test_prediction"] - future) <= v2_widths(frame, fitted, target, rows)
    return {
        "rows": int(len(rows)),
        "subjects": int(frame.iloc[rows]["subject_id"].nunique()),
        "coverage": group_equal_coverage(frame, rows, covered, "subject_id"),
        "median_current_value": float(np.median(current)),
        "p90_abs_change": float(np.percentile(np.abs(future - current), 90)),
    }


def run(args: argparse.Namespace) -> dict:
    development = {row["cell"]: row for row in json.loads(args.development_report.read_text(encoding="utf-8"))["cells"]}
    cells = []
    for cell in args.cells.split(","):
        contract = development[cell]
        target, horizon, source = contract["target"], int(contract["horizon_hours"]), contract["source_model"]
        raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        mimic, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=15000)
        del raw
        modules = {}
        for module in MODULES:
            path = eicu_path(args.eicu_root, module, horizon)
            if not path.exists():
                continue
            eicu, _ = normalize_cohort(pd.read_parquet(path), database="eICU", target=target, horizon=horizon, max_rows=15000)
            if len(eicu) < MIN_ROWS or eicu["subject_id"].nunique() < MIN_SUBJECTS:
                continue
            print(f"{cell} on {module}", flush=True)
            modules[module] = score(mimic, eicu, cell=cell, target=target, source=source)
        cells.append({"cell": cell, "modules": modules})
    output = {"schema": "eicu_module_cohort_diagnostic.v1", "cells": cells, "causal_claim_allowed": False}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--cells", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    for cell in run(parse_args())["cells"]:
        print(cell["cell"])
        for module, item in sorted(cell["modules"].items(), key=lambda kv: kv[1]["coverage"]):
            print(f"   {module:28s} rows {item['rows']:6d} cov {item['coverage']:.3f}  median value {item['median_current_value']:.2f}  p90 |change| {item['p90_abs_change']:.2f}")
