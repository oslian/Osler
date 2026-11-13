"""Select each cell's point recipe on the external development databases.

Training is MIMIC-IV only (v4 data): AGENTS.md multi-sample rule, 5 patient
samples of 15,000 rows, sample 0 = the original audit sample.  Development
test databases: eICU (harmonized), NWICU (v4) and Zigong (laboratory only),
each capped at 60,000 rows as in ``site_adaptive_external_audit``.  All three
were already used, so this is development selection, not validation; the
untouched SICdb / HiRID stay reserved for a frozen lockbox.

Every candidate recipe (``recipe_shift_selection_dev.recipes_for``) is
deployed in the frozen form (online point recalibration ``affine_0.005`` +
day-one weighting + DtACI, one stream per hospital) in its deployable form at
each database (``osler_jepa.recipe_availability``), and scored with its own
comparator gate (MPR vs persistence; TCR and plain ridge vs MPR; change
hurdle vs plain ridge) on the database's gate axes.

Predeclared rule (written before any candidate was scored), per cell:

1. point-robust at a database = point gate passed in >= 80% of samples;
2. select the recipe point-robust at the most evaluable databases; ties go
   to the simplest recipe in the order MPR, TCR, plain ridge, hurdle;
3. persistence fallback: if the selected recipe beats persistence in >= 80%
   of samples at fewer than a strict majority of evaluable databases, the
   cell's point output is persistence.

Interval calibration is not part of the selection.  Factual research-only.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from external_value_conditional_interval_audit import GATE_AXES, evaluability, interval_report
from mimic_to_eicu_49_cell_external_audit import (
    choose_eicu_cohort,
    fit_continuous_recipe,
    normalize_cohort,
    paired_comparison,
    stable_seed,
)
from multi_sample_external_audit import mimic_sample
from osler_jepa.recipe_availability import deployable_recipe
from recipe_shift_selection_dev import RECIPES, recipes_for
from site_adaptive_external_audit import DTACI_METHOD, frozen_dtaci_config, frozen_point_config, scored_only, simulate_sites
from target_specific_domain_conformal_audit import _safe_float


DATABASES = ("eICU", "NWICU", "Zigong")
PREFIX = {"eICU": "EICU:", "NWICU": "NWICU:", "Zigong": "ZIGONG:"}
SOURCE_ROWS = 15000
TEST_ROWS = 60000
ROBUST_SHARE = 0.8


def load_cohorts(args: argparse.Namespace, raw: dict[str, pd.DataFrame], target: str, horizon: int) -> dict[str, pd.DataFrame]:
    cohorts = {"MIMIC-IV": normalize_cohort(raw["MIMIC-IV"], database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)[0]}
    cohorts["eICU"] = choose_eicu_cohort(args.eicu_root, target=target, horizon=horizon, max_rows=TEST_ROWS)[0]
    for database in ("NWICU", "Zigong"):
        cohorts[database] = normalize_cohort(raw[database], database=database, target=target, horizon=horizon, max_rows=TEST_ROWS)[0]
    return cohorts


def deploy(train: pd.DataFrame, test: pd.DataFrame, *, database: str, cell: str, target: str, horizon: int,
           recipe: str, sample: int, point_config, dtaci_config) -> dict:
    frame = pd.concat([train, test], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    subject = frame["subject_id"].astype(str)
    train_rows = np.flatnonzero(subject.str.startswith("MIMIC:"))
    test_rows = np.flatnonzero(subject.str.startswith(PREFIX[database]))
    seed = stable_seed(cell, "mimic_to_eicu")
    fitted = fit_continuous_recipe(frame, train_rows, test_rows, target=target, source=recipe, seed=seed)
    rows = fitted["test_rows"]
    rows, values = scored_only(rows, simulate_sites(frame, rows, fitted, target, horizon, cell, sample, point_config,
                                                    dtaci_config)[DTACI_METHOD])
    observed = _safe_float(frame, f"future_{target}")[rows]
    current = _safe_float(frame, f"{target}_t")[rows]
    loss = np.abs(values["prediction"] - observed)
    axes = GATE_AXES[database]
    persistence = paired_comparison(frame, rows, loss, np.abs(current - observed), seed=seed)
    comparator = paired_comparison(frame, rows, loss, np.abs(values["comparator"] - observed), seed=seed + 2)
    beats_persistence = all(persistence[axis]["pass"] for axis in axes["point"])
    point = beats_persistence and all(comparator[axis]["pass"] for axis in axes["point"])
    report = interval_report(frame, rows, loss <= values["width"], values["width"], axes["interval"])
    return {
        "point_pass": bool(point),
        "beats_persistence": bool(beats_persistence),
        "full_pass": bool(point and report["pass"]),
        "mae_over_persistence": float(loss.mean() / max(np.abs(current - observed).mean(), 1e-9)),
    }


def summarize(samples: list[dict]) -> dict:
    shares = {key: float(np.mean([s[key] for s in samples])) for key in ("point_pass", "beats_persistence", "full_pass")}
    return {
        **{f"{key}_share": value for key, value in shares.items()},
        "point_robust": shares["point_pass"] >= ROBUST_SHARE,
        "persistence_robust": shares["beats_persistence"] >= ROBUST_SHARE,
        "full_robust": shares["full_pass"] >= ROBUST_SHARE,
        "mae_over_persistence": float(np.mean([s["mae_over_persistence"] for s in samples])),
        "per_sample": samples,
    }


def select(candidates: dict[str, dict[str, dict]], databases: list[str]) -> dict:
    """Apply the predeclared rule to {recipe: {database: summary}}."""

    order = [recipe for recipe in RECIPES if recipe in candidates]
    scores = {recipe: sum(candidates[recipe][d]["point_robust"] for d in databases) for recipe in order}
    selected = max(order, key=lambda recipe: (scores[recipe], -order.index(recipe)))
    beats = sum(candidates[selected][d]["persistence_robust"] for d in databases)
    return {
        "point_robust_counts": scores,
        "selected": selected,
        "selected_beats_persistence_databases": beats,
        "persistence_fallback": bool(2 * beats <= len(databases)),
    }


def audit_horizon(args: argparse.Namespace, contracts: list[dict]) -> list[dict]:
    """All cells of one horizon; one process per horizon keeps one set of raw tables in memory."""

    _, point_config = frozen_point_config(args.point_dev)
    _, dtaci_config = frozen_dtaci_config(args.dtaci_dev)
    horizon = int(contracts[0]["horizon_hours"])
    raw = {
        "MIMIC-IV": pd.read_parquet(args.mimic_template.format(horizon=horizon)),
        "NWICU": pd.read_parquet(args.nwicu_template.format(horizon=horizon)),
        "Zigong": pd.read_parquet(args.zigong_template.format(horizon=horizon)),
    }
    cells = []
    for contract in contracts:
        cell, target = contract["cell"], contract["target"]
        cohorts = load_cohorts(args, raw, target, horizon)
        databases = [d for d in DATABASES if evaluability(d, cohorts[d]) is None]
        trains = [mimic_sample(cohorts["MIMIC-IV"], target=target, horizon=horizon, sample=s, rows=SOURCE_ROWS)
                  for s in range(args.samples)]
        candidates: dict[str, dict[str, dict]] = {}
        scored: dict[tuple[str, str], dict] = {}
        for recipe in recipes_for(target):
            candidates[recipe] = {}
            for database in databases:
                deployed, _ = deployable_recipe(recipe, cohorts[database], target)
                if (deployed, database) not in scored:
                    scored[deployed, database] = {"deployed": deployed, **summarize([
                        deploy(train, cohorts[database], database=database, cell=cell, target=target, horizon=horizon,
                               recipe=deployed, sample=sample, point_config=point_config, dtaci_config=dtaci_config)
                        for sample, train in enumerate(trains)
                    ])}
                candidates[recipe][database] = scored[deployed, database]
        entry = {"cell": cell, "catalog_recipe": contract["source_model"], "databases": databases,
                 "candidates": candidates}
        if databases:
            entry.update(select(candidates, databases))
        print(json.dumps({"cell": cell, **{k: entry.get(k) for k in ("point_robust_counts", "selected", "persistence_fallback")}}), flush=True)
        cells.append(entry)
    return cells


def run(args: argparse.Namespace) -> dict:
    development = json.loads(args.development_report.read_text(encoding="utf-8"))
    contracts = sorted((r for r in development["cells"] if r["task_type"] != "binary_event_probability"),
                       key=lambda r: (int(r["horizon_hours"]), r["target"]))
    if args.cells:
        contracts = [r for r in contracts if r["cell"] in set(args.cells.split(","))]
    groups = [[r for r in contracts if int(r["horizon_hours"]) == h]
              for h in sorted({int(r["horizon_hours"]) for r in contracts})]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cells = [cell for group in pool.map(audit_horizon, [args] * len(groups), groups) for cell in group]
    selected = [c for c in cells if "selected" in c]
    output = {
        "schema": "external_recipe_selection_dev.v1",
        "training": "MIMIC-IV only, 5 samples x 15,000 rows",
        "development_databases": list(DATABASES),
        "selection_rule": "most databases point-robust, ties to simplest; persistence fallback when the selected "
                          "recipe beats persistence robustly at no strict majority of evaluable databases",
        "recipes": {c["cell"]: "persistence" if c["persistence_fallback"] else c["selected"] for c in selected},
        "persistence_fallback": sorted(c["cell"] for c in selected if c["persistence_fallback"]),
        "changed_from_catalog": sorted(c["cell"] for c in selected if c["selected"] != c["catalog_recipe"]),
        "cells": cells,
        "independent_validation": False,
        "causal_claim_allowed": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-report", type=Path, default=Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json"))
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet")
    parser.add_argument("--eicu-root", type=Path, default=Path("eicu_harmonized_20261022"))
    parser.add_argument("--nwicu-template", default="nwicu_observation_transitions_{horizon}h_v5.parquet")
    parser.add_argument("--zigong-template", default="zigong_observation_transitions_{horizon}h_v2.parquet")
    parser.add_argument("--point-dev", type=Path, default=Path("site_adaptive_point_dev_20261016.json"))
    parser.add_argument("--dtaci-dev", type=Path, default=Path("site_adaptive_dtaci_dev_20261017.json"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cells", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({k: result[k] for k in ("recipes", "persistence_fallback", "changed_from_catalog")}, indent=1))
