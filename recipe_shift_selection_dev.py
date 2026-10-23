"""Re-select each cell's recipe by robustness under simulated MIMIC deployment shift.

The 49-cell catalog chose recipes by in-distribution MIMIC gates; externally the
extra structure of several recipes (treatment context, change hurdle) beats
persistence but not its simpler comparator.  Here every candidate recipe is
deployed in the MIMIC S1 (time) and S2 (cardiac units) scenarios with the frozen
deployment form (online point recalibration ``affine_0.005`` + day-one
weighting + DtACI meta rate 20) and scored with its own original gate:

- measurement_pure_ridge vs persistence;
- observed_treatment_context_ridge and plain_ridge vs measurement_pure_ridge;
- target_specific_change_hurdle vs plain_ridge (targets with a change threshold).

Predeclared selection rule (written before any recipe was scored), per cell:
the simplest recipe (order above) that passes the full gate in both
scenarios; if none passes both, the simplest recipe passing the most
scenarios.  No eICU or NWICU data is read.  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import group_equal_coverage, normalize_cohort, paired_comparison
from osler_jepa.site_adaptive_conformal import DtaciConfig, dtaci_stream_widths
from site_adaptive_interval_dev import BOUNDS
from site_adaptive_point_dev import SCENARIOS, candidates as point_candidates, corrected_experts, scenario_streams
from target_specific_domain_conformal_audit import TARGET_CHANGE_THRESHOLDS


RECIPES = ("measurement_pure_ridge", "observed_treatment_context_ridge", "plain_ridge", "target_specific_change_hurdle")
POINT_CONFIG = point_candidates()["affine_0.005"]
DTACI_CONFIG = DtaciConfig(meta_rate=20.0)


def recipes_for(target: str) -> tuple[str, ...]:
    return tuple(r for r in RECIPES if r != "target_specific_change_hurdle" or target in TARGET_CHANGE_THRESHOLDS)


def score(stream: dict) -> dict:
    frame, rows, observed, now = stream["frame"], stream["rows"], stream["observed"], stream["now"]
    corrected, comparator = corrected_experts(stream, POINT_CONFIG)
    loss = np.abs(corrected - observed)
    point = all(
        paired_comparison(frame, rows, loss, other, seed=stream["seed"])["patient"]["pass"]
        for other in (np.abs(now - observed), np.abs(comparator - observed))
    )
    widths = dtaci_stream_widths(calibration_scores=stream["cal_scores"], calibration_weights=stream["weights"],
                                 scales=stream["scales"], predictions=corrected, observed=observed,
                                 config=DTACI_CONFIG, **stream["common"]).widths
    covered = loss <= widths
    patient = group_equal_coverage(frame, rows, covered, "subject_id")
    unit = group_equal_coverage(frame, rows, covered, "careunit")
    return {
        "point_pass": bool(point),
        "patient_equal_coverage": patient,
        "careunit_equal_coverage": unit,
        "full_pass": bool(point and BOUNDS[0] <= patient <= BOUNDS[1] and BOUNDS[0] <= unit <= BOUNDS[1]),
        "mae": float(loss.mean()),
    }


def select(results: dict[str, dict[str, dict]], recipes: tuple[str, ...]) -> str:
    passes = {recipe: sum(results[recipe][s]["full_pass"] for s in SCENARIOS) for recipe in recipes}
    best = max(passes.values())
    return next(recipe for recipe in recipes if passes[recipe] == best)


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = [row for row in development["cells"] if row["task_type"] != "binary_event_probability"]
    patients = pd.read_csv(args.patients, usecols=["subject_id", "anchor_year_group"])
    years = pd.Series(patients["anchor_year_group"].to_numpy(), index="MIMIC:" + patients["subject_id"].astype(str))
    by_horizon: dict[int, list[dict]] = {}
    for row in contracts:
        by_horizon.setdefault(int(row["horizon_hours"]), []).append(row)
    cells = []
    for horizon in sorted(by_horizon):
        raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        for contract in sorted(by_horizon[horizon], key=lambda row: row["target"]):
            cell, target = contract["cell"], contract["target"]
            print(f"recipe shift selection {cell}", flush=True)
            full, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
            full = full.assign(first_careunit=full["careunit"])
            recipes = recipes_for(target)
            results = {
                recipe: {name: score(stream) for name, stream in scenario_streams(
                    full, years, cell=cell, target=target, horizon=horizon, source=recipe)}
                for recipe in recipes
            }
            selected = select(results, recipes)
            cells.append({
                "cell": cell,
                "catalog_recipe": contract["source_model"],
                "selected_recipe": selected,
                "changed": selected != contract["source_model"],
                "results": results,
            })
            print(json.dumps({"cell": cell, "catalog": contract["source_model"], "selected": selected}), flush=True)
        del raw
    output = {
        "schema": "recipe_shift_selection_dev.v1",
        "database": "MIMIC-IV only",
        "selection_rule": "simplest recipe passing the full gate in both MIMIC shift scenarios; else simplest with most passes",
        "deployment_form": "affine_0.005 point recalibration + day-one weighting + DtACI meta rate 20",
        "recipes": {cell["cell"]: cell["selected_recipe"] for cell in cells},
        "changed_cells": [cell["cell"] for cell in cells if cell["changed"]],
        "cells": cells,
        "external_data_read": False,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-report", type=Path, default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"))
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet")
    parser.add_argument("--patients", type=Path, default=Path.home() / "Downloads/mimic-iv-3.1/hosp/patients.csv.gz")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"changed": result["changed_cells"]}, indent=1))
