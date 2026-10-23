"""Cross-database audit of the sparse-aware architecture on MIMIC-IV DKA.

This is an external database audit of the recipe, not a claim that an eICU
checkpoint transfers unchanged. MIMIC-IV DKA is the external transition
contract currently available locally; other body modules are reported as
unavailable rather than silently mapped to DKA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nonlinear_latent_rollout import _history_indices, _time_hours, state_variables
from sparse_aware_pure_jepa import _split_subjects, evaluate, fit_one

SEEDS = (7, 11, 19, 23, 37, 53, 71)


def load_mimic_dka(path: str | Path):
    frame = pd.read_parquet(path).copy()
    variables = tuple(sorted(state_variables(frame)))
    for variable in variables:
        source = f"{variable}_tp6"
        if source in frame:
            frame.rename(columns={source: f"future_{variable}"}, inplace=True)
        elif f"{variable}_t" in frame:
            frame[f"future_{variable}"] = np.nan
    frame["_source_horizon"] = 6
    frame["hospitalid"] = "MIMIC-IV"
    frame["_time_hours"] = _time_hours(frame)
    frame.sort_values(["stay_id", "_time_hours"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    history = _history_indices(frame, history=5)
    for position in range(history.shape[1]):
        frame[f"_history_{position}"] = history[:, position]
    return frame, variables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dka_transitions_6h_mimiciv_full_v31_icd.parquet")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", default="mimiciv_sparse_aware_external_audit.json")
    args = parser.parse_args()
    frame, variables = load_mimic_dka(args.data)
    reports = {}
    for seed in SEEDS:
        train, test = _split_subjects(frame, seed)
        model, arrays, losses = fit_one(frame, variables, train, seed=seed, epochs=args.epochs, batch_size=args.batch_size)
        metrics = evaluate(model, arrays, frame, variables, test)
        reports[str(seed)] = {
            "train_subjects": int(frame.iloc[train].subject_id.nunique()),
            "test_subjects": int(frame.iloc[test].subject_id.nunique()),
            "loss_history": losses,
            "metrics": metrics,
        }
    summary = {}
    for target in variables:
        cells = [r["metrics"].get(target, {}) for r in reports.values()]
        cells = [c for c in cells if c.get("candidate_mae") is not None]
        summary[target] = {
            "seeds": len(cells),
            "mean_delta_vs_persistence": float(np.mean([c["delta_vs_persistence"] for c in cells])) if cells else None,
            "significant_bootstrap_not_run": True,
        }
    result = {
        "schema": "mimiciv_sparse_aware_external_audit.v1",
        "database": "MIMIC-IV",
        "cohort": "ICD-supported DKA transition parquet",
        "rows": int(len(frame)),
        "subjects": int(frame.subject_id.nunique()),
        "variables": list(variables),
        "patient_heldout": reports,
        "summary": summary,
        "hospital_gate": {"status": "not_available", "reason": "MIMIC-IV transition artifact has no hospital identifier"},
        "promotion_status": "external_research_only",
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "boundary": "This validates the sparse-aware architecture recipe on an independent database, not transfer of an eICU checkpoint and not all body modules.",
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved {args.output}; MIMIC rows={len(frame)} subjects={frame.subject_id.nunique()}", flush=True)


if __name__ == "__main__":
    main()
