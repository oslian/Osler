"""Independent hospital-heldout evidence for the final sparse-aware artifacts.

The final full-cohort checkpoint is fit after target selection.  This script
does not score that same full-fit checkpoint on its training rows.  Instead it
re-trains the same architecture on all non-heldout hospitals, calibrates on a
patient split inside that training set, and evaluates on a completely heldout
hospital.  The resulting evidence can unlock research-only runtime movement;
it never grants causal or clinical authority.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from nonlinear_latent_rollout import load_multihorizon_cohort
from sparse_aware_full_gate import MODULES, _evaluate_gate, _split_groups
from sparse_aware_pure_jepa import _split_subjects, conformal_calibration, fit_one


def audit_module(data_root, module, horizons, targets, epochs, batch_size, max_stays, seed):
    frame, variables = load_multihorizon_cohort(
        data_root, module, horizons, max_stays_per_horizon=max_stays,
        history=5, seed=seed,
    )
    if "hospitalid" not in frame or frame["hospitalid"].nunique() < 4:
        return {"status": "not_available", "error": "fewer than four hospitals"}
    train, test = _split_groups(frame, "hospitalid", seed)
    train_subject, calibration_local = _split_subjects(
        frame.iloc[train].reset_index(drop=True), seed + 1001, 0.20
    )
    calibration = train[calibration_local]
    fit_rows = train[train_subject]
    model, arrays, losses = fit_one(
        frame, variables, fit_rows, seed=seed, epochs=epochs, batch_size=batch_size
    )
    metrics = _evaluate_gate(
        model, arrays, frame, variables, test, seed, baseline_rows=fit_rows
    )
    conformal = conformal_calibration(model, arrays, variables, calibration, test)
    target_report = {}
    for target in targets:
        value = metrics.get(target, {})
        interval = conformal.get(target, {})
        scoped = interval.get("by_horizon", {})
        interval_cells = [cell for cell in scoped.values() if cell.get("status") in {"PASS", "REJECT"}]
        interval_ok = bool(
            interval_cells
            and all(cell.get("status") == "PASS" for cell in interval_cells)
        )
        value_ok = bool(value.get("significant_vs_persistence", False))
        target_report[target] = {
            "pass": value_ok and interval_ok,
            "value_gate": value_ok,
            "conformal_gate": interval_ok,
            "value": value,
            "conformal": interval,
            "observation_process": metrics.get("_measurement_process", {}).get(target, {}),
        }
    result = {
        "status": "run",
        "module": module,
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "hospital_count": int(frame["hospitalid"].nunique()),
        "train_rows": int(len(fit_rows)),
        "heldout_rows": int(len(test)),
        "heldout_hospital_ids": sorted(str(value) for value in frame.iloc[test]["hospitalid"].dropna().unique()),
        "loss_history": losses,
        "targets": target_report,
        "boundary": {
            "research_only": True,
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
    }
    del model, arrays, frame
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--full-gate", default="sparse_aware_full_gate.json")
    parser.add_argument("--modules", default=",".join(MODULES))
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-stays", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", default="sparse_aware_independent_holdout.json")
    args = parser.parse_args()
    full_gate = json.loads(Path(args.full_gate).read_text(encoding="utf-8"))
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    max_stays = None if args.max_stays < 0 else args.max_stays
    modules = {}
    for module in [value.strip() for value in args.modules.split(",") if value.strip()]:
        targets = [
            target for target, metrics in full_gate.get(module, {}).get("aggregate", {}).items()
            if metrics.get("validated_full_gate")
        ]
        if not targets:
            continue
        print(f"independent hospital-heldout audit: {module} ({len(targets)} targets)", flush=True)
        try:
            modules[module] = audit_module(
                args.data_root, module, horizons, targets,
                args.epochs, args.batch_size, max_stays, args.seed,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            modules[module] = {"status": "not_available", "error": str(exc)}
    output = {
        "schema": "sparse_aware_independent_hospital_holdout.v1",
        "modules": modules,
        "policy": {
            "unvalidated_target_action": "persistence",
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    passed = sum(
        int(cell.get("pass", False))
        for module in modules.values() if module.get("status") == "run"
        for cell in module.get("targets", {}).values()
    )
    print(f"saved {args.output}; independent target passes: {passed}", flush=True)


if __name__ == "__main__":
    main()
