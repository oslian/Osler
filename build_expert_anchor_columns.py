"""Attach frozen expert-cell predictions to JEPA event examples as system-layer anchors.

AGENTS.md: validated target x horizon expert cells are projected onto the system
layer as forecast anchors and the hierarchy learns a zero-initialized residual.
For every event-example row and every JEPA target, ``expert__anchor_<target>`` (the loader keeps ``expert__`` columns; they are anchors, never input features)
holds the frozen deployment recipe's prediction for that row's horizon:

- recipe per cell from ``deployment_recipes_20261020.json`` (catalog cells);
  cells outside the catalog use persistence (the current value);
- fitted exactly as deployed: MIMIC-IV v5 transitions (all anchors kept), training sample 0
  (15,000 rows, ``multi_sample_external_audit.mimic_sample``), seed
  ``stable_seed(cell, "mimic_to_eicu")``;
- applied to event rows of every database (MIMIC and external rows alike get
  the static MIMIC-fitted prediction; site online correction is applied later,
  in the deployment-form evaluation);
- every row with a current value gets the frozen recipe's prediction, whether or
  not a future label exists (forecast-time invariance); histories come from the
  complete event stream; the recipe is fitted on MIMIC patients outside the JEPA
  cohort, so anchors are out-of-sample for all JEPA rows.

Writes the augmented examples and an ``--expert-anchor-registry`` JSON for
``teacher_anchored_joint_jepa_audit.py``.  Research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimic_to_eicu_49_cell_external_audit import fit_continuous_recipe, normalize_cohort, stable_seed
from target_specific_domain_conformal_audit import add_history_features
from multi_sample_external_audit import mimic_sample


SOURCE_ROWS = 15000


def anchor_predictions(events: pd.DataFrame, mimic_raw: pd.DataFrame, target: str, horizon: int, recipe: str,
                       exclude_subjects: set[str]) -> pd.Series:
    """Frozen recipe predictions for every event row of one horizon that has a current value.

    Forecast-time invariant: histories come from the complete event stream and no
    row's anchor depends on whether a future label exists.  The recipe is fitted on
    MIMIC-IV transitions excluding ``exclude_subjects`` (the JEPA cohort's MIMIC
    patients), so anchors are out-of-sample for every JEPA fitting, selection and
    calibration row.
    """

    current = pd.to_numeric(events[f"{target}_t"], errors="coerce")
    anchor = pd.Series(np.nan, index=events.index)
    if recipe == "persistence":
        return current
    full = normalize_cohort(mimic_raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)[0]
    full = full[~full["subject_id"].str.removeprefix("MIMIC:").isin(exclude_subjects)].reset_index(drop=True)
    train = mimic_sample(full, target=target, horizon=horizon, sample=0, rows=SOURCE_ROWS)
    rows = events.copy()
    rows["_event_index"] = rows.index
    rows["subject_id"] = "EVENT:" + rows["subject_id"].astype(str)
    rows["stay_id"] = "EVENT:" + rows["database"].astype(str) + ":" + rows["stay_id"].astype(str)
    rows["anchor_time"] = pd.to_datetime(rows["anchor_time"])
    rows = add_history_features(rows)                      # complete event stream, before anything else
    rows = rows[current.notna()]
    frame = pd.concat([train, rows], ignore_index=True, sort=False)
    frame.sort_values(["subject_id", "anchor_time"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    subject = frame["subject_id"].astype(str)
    fitted = fit_continuous_recipe(frame, np.flatnonzero(subject.str.startswith("MIMIC:")),
                                   np.flatnonzero(subject.str.startswith("EVENT:")), target=target, source=recipe,
                                   seed=stable_seed(f"{target}@{horizon}h", "mimic_to_eicu"), require_test_labels=False)
    predicted = pd.Series(fitted["test_prediction"], index=frame.loc[fitted["test_rows"], "_event_index"].astype(int).to_numpy())
    anchor.loc[predicted.index] = predicted.to_numpy()
    return anchor.fillna(current)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--recipes", type=Path, default=Path("deployment_recipes_20261020.json"))
    parser.add_argument("--mimic-template", default="mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe_v5.parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    recipes = json.loads(args.recipes.read_text(encoding="utf-8"))["recipes"]
    examples = pd.read_parquet(args.examples)
    targets = args.targets.split(",")
    jepa_mimic_subjects = set(examples.loc[examples["database"].astype(str).str.lower() == "mimiciv", "subject_id"].astype(str))
    for target in targets:
        examples[f"expert__anchor_{target}"] = pd.to_numeric(examples[f"{target}_t"], errors="coerce")
    cells = []
    for horizon in sorted(int(h) for h in examples["horizon_hours"].unique()):
        mimic_raw = pd.read_parquet(args.mimic_template.format(horizon=horizon))
        events = examples[examples["horizon_hours"] == horizon]
        for target in targets:
            cell = f"{target}@{horizon}h"
            recipe = recipes.get(cell, "persistence")
            examples.loc[events.index, f"expert__anchor_{target}"] = anchor_predictions(events, mimic_raw, target, horizon, recipe,
                                                                                         jepa_mimic_subjects)
            cells.append({"cell": cell, "target": target, "horizon_hours": horizon, "task_type": "continuous_value",
                          "anchor_column": f"expert__anchor_{target}", "source_model": recipe})
            print(json.dumps({"cell": cell, "recipe": recipe}), flush=True)
    examples.to_parquet(args.output, index=False)
    args.registry.write_text(json.dumps({
        "schema": "expert_anchor_registry.v1",
        "source": "frozen deployment recipes fitted on MIMIC-IV v5 sample 0 (JEPA-cohort subjects excluded); static (no site online correction)",
        "recipes": str(args.recipes), "cells": cells, "causal_claim_allowed": False,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
