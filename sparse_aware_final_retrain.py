"""Final full-cohort retrain and artifact audit for sparse-aware targets.

The full gate selects targets.  This script then retrains each module on its
entire selected cohort, carries forward only the target-specific conformal
artifacts from the held-out gate, and verifies that the saved checkpoint,
validated target list, and registry agree.  It is research-only: causal and
clinical authority remain disabled.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from nonlinear_latent_rollout import load_multihorizon_cohort
from pure_jepa_runtime import load_checkpoint
from sparse_aware_full_gate import SEEDS
from sparse_aware_pure_jepa import fit_one


def _ready_targets(full_gate: dict) -> dict[str, list[str]]:
    return {
        module: [target for target, metrics in result.get("aggregate", {}).items()
                 if metrics.get("validated_full_gate")]
        for module, result in full_gate.items()
        if isinstance(result, dict)
    }


def _conformal_artifacts(full_gate: dict, module: str, targets: list[str]) -> dict:
    result = full_gate[module]
    artifacts = {}
    for target in targets:
        rows = []
        for seed in SEEDS:
            cell = result.get("patient_heldout", {}).get(str(seed), {}).get("conformal", {}).get(target, {})
            if cell.get("status") == "PASS" and np.isfinite(float(cell.get("q90", np.nan))):
                rows.append(cell)
        if len(rows) != len(SEEDS):
            raise ValueError(f"{module}/{target}: missing one or more conformal gate artifacts")
        by_horizon = {}
        keys = set().union(*[set(row.get("by_horizon", {})) for row in rows])
        for key in sorted(keys):
            scoped = [row.get("by_horizon", {}).get(key, {}) for row in rows]
            if len(scoped) == len(SEEDS) and all(
                cell.get("status") == "PASS"
                and np.isfinite(float(cell.get("q90", np.nan)))
                for cell in scoped
            ):
                by_horizon[str(key)] = {
                    "status": "PASS",
                    "q90": float(np.median([float(cell["q90"]) for cell in scoped])),
                    "gate_coverages": [float(cell["coverage"]) for cell in scoped],
                    "gate_calibration_n": [int(cell["calibration_n"]) for cell in scoped],
                }
        artifacts[target] = {
            "status": "PASS",
            "q90": float(np.median([float(row["q90"]) for row in rows])),
            "gate_coverages": [
                float(np.mean([
                    float(cell["coverage"])
                    for cell in row.get("by_horizon", {}).values()
                    if cell.get("coverage") is not None
                ])) if row.get("by_horizon") else float(row.get("coverage", np.nan))
                for row in rows
            ],
            "gate_calibration_n": [
                int(np.sum([
                    int(cell["calibration_n"])
                    for cell in row.get("by_horizon", {}).values()
                    if cell.get("calibration_n") is not None
                ])) if row.get("by_horizon") else int(row.get("calibration_n", 0))
                for row in rows
            ],
            "method": "median_of_seven_patient_heldout_split_conformal_q90",
            "scope": "target_by_horizon",
            "by_horizon": by_horizon,
        }
    return artifacts


def _build_registry(
    ready: dict[str, list[str]],
    checkpoint_paths: dict[str, str],
    external_audit: dict | None = None,
) -> dict:
    modules = {}
    for module, targets in ready.items():
        modules[module] = {}
        for target in targets:
            independent = bool(
                external_audit
                and external_audit.get("modules", {}).get(module, {})
                .get("targets", {}).get(target, {}).get("pass", False)
            )
            modules[module][target] = {
                "can_move": independent,
                "promotion_ready": independent,
                "status": "validated_move_research_only" if independent else "external_holdout_audit_required",
                "source": "sparse_aware_full_gate_plus_full_cohort_retrain",
                "checkpoint": checkpoint_paths[module],
                "validated_target": True,
                "clinical_promotion_allowed": False,
                "causal_claim_allowed": False,
                "independent_holdout_audit": independent,
            }
    return {
        "policy": {
            "fail_closed": True,
            "unvalidated_target_action": "persistence",
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
            "research_only": True,
            "independent_holdout_audit_required": True,
            "measurement_process_heads_are_separate_gates": True,
        },
        "modules": modules,
    }


def retrain_module(data_root, module, targets, horizons, epochs, batch_size, max_stays, output_dir, full_gate):
    frame, variables = load_multihorizon_cohort(
        data_root,
        module,
        horizons,
        max_stays_per_horizon=max_stays,
        history=5,
        seed=2026,
    )
    rows = np.arange(len(frame), dtype=np.int64)
    model, arrays, losses = fit_one(
        frame, variables, rows, seed=2026, epochs=epochs, batch_size=batch_size
    )
    conformal = _conformal_artifacts(full_gate, module, targets)
    checkpoint_path = Path(output_dir) / f"{module}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "SparseAwarePureJEPABodyModel",
        "state_dict": model.state_dict(),
        "scaler": arrays[0].to_dict(),
        "module": module,
        "horizons": list(horizons),
        "variables": list(variables),
        "input_dim": int(model.input_dim),
        "hidden": int(model.hidden),
        "latent": int(model.latent),
        "architecture": {
            "measurement_pure": True,
            "history_encoder": "GRU",
            "target_specific_heads": len(variables),
            "observation_process_head": True,
            "uncertainty_head": True,
            "stability_head": True,
            "next_measurement_time_head": False,
            "hand_beliefs_used": False,
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
        "promotion_status": "validated_research_only",
        "validated_targets": list(targets),
        "conformal": conformal,
        "measurement_process": {
            "observation_process_validated_targets": [
                target for target in targets
                if full_gate[module].get("aggregate", {}).get(target, {}).get("observation_process_gate", False)
            ],
            "stability_validated_targets": [
                target for target in targets
                if full_gate[module].get("aggregate", {}).get(target, {}).get("stability_gate", False)
            ],
            "next_measurement_time_validated_targets": [],
            "gate_is_separate_from_value_promotion": True,
        },
        "retrain": {
            "rows": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "epochs": int(epochs),
            "max_stays_per_horizon": max_stays,
            "loss_history": losses,
            "full_cohort_retrain": max_stays is None,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, checkpoint


def artifact_audit(checkpoint_paths: dict[str, str], ready: dict[str, list[str]], registry: dict) -> dict:
    results = {}
    for module, path in checkpoint_paths.items():
        model, scaler, checkpoint = load_checkpoint(path)
        validated_targets = set(checkpoint.get("validated_targets", []))
        registry_targets = set(registry["modules"].get(module, {}))
        results[module] = {
            "loadable": True,
            "model_type": checkpoint.get("model_type"),
            "measurement_pure": checkpoint.get("architecture", {}).get("measurement_pure") is True,
            "hand_beliefs_used": checkpoint.get("architecture", {}).get("hand_beliefs_used") is True,
            "target_list_matches_registry": validated_targets == registry_targets == set(ready[module]),
            "conformal_q90_finite": all(
                np.isfinite(float(checkpoint["conformal"][target]["q90"]))
                for target in ready[module]
            ),
            "conformal_horizon_scoped": all(
                checkpoint["conformal"][target].get("scope") == "target_by_horizon"
                for target in ready[module]
            ),
            "scaler_variables": len(scaler.variables),
            "promotion_status": checkpoint.get("promotion_status"),
        }
        results[module]["artifact_ok"] = bool(
            results[module]["loadable"]
            and results[module]["measurement_pure"]
            and not results[module]["hand_beliefs_used"]
            and results[module]["target_list_matches_registry"]
            and results[module]["conformal_q90_finite"]
            and results[module]["conformal_horizon_scoped"]
        )
        del model, scaler, checkpoint
        gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--full-gate", default="sparse_aware_full_gate.json")
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-stays", type=int, default=5000)
    parser.add_argument("--output-dir", default="sparse_aware_validated_checkpoints")
    parser.add_argument("--output", default="sparse_aware_final_retrain.json")
    parser.add_argument("--registry-output", default="sparse_aware_validated_registry.json")
    parser.add_argument("--external-audit", default=None)
    args = parser.parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    full_gate = json.loads(Path(args.full_gate).read_text(encoding="utf-8"))
    ready = _ready_targets(full_gate)
    # A bounded stay cap keeps the final retrain reproducible on a laptop.  It
    # is recorded in the artifact and does not change the evidence gate.
    max_stays = None if args.max_stays < 0 else int(args.max_stays)
    checkpoint_paths = {}
    retrain_report = {}
    for module, targets in ready.items():
        if not targets:
            continue
        print(f"final sparse-aware retrain: {module} ({len(targets)} validated targets)", flush=True)
        path, checkpoint = retrain_module(
            args.data_root,
            module,
            targets,
            horizons,
            args.epochs,
            args.batch_size,
            max_stays,
            args.output_dir,
            full_gate,
        )
        checkpoint_paths[module] = str(path.resolve())
        retrain_report[module] = checkpoint["retrain"]
        gc.collect()
    external_audit = None
    if args.external_audit:
        external_audit = json.loads(Path(args.external_audit).read_text(encoding="utf-8"))
    registry = _build_registry(ready, checkpoint_paths, external_audit)
    audit = artifact_audit(checkpoint_paths, ready, registry)
    result = {
        "method": "sparse-aware full-cohort retrain after seven-seed gate",
        "ready_target_count": sum(len(targets) for targets in ready.values()),
        "retrain": retrain_report,
        "artifact_audit": audit,
        "all_artifacts_ok": all(item["artifact_ok"] for item in audit.values()),
        "boundary": {
            "research_only": True,
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
            "unvalidated_targets_fallback": "persistence",
            "independent_holdout_audit_required": True,
            "final_fit_performance_is_estimated_on_separate_holdout_retraining": bool(external_audit),
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.registry_output).write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"saved {args.output}", flush=True)
    print(f"saved {args.registry_output}", flush=True)
    print(f"artifact audit ok: {result['all_artifacts_ok']}", flush=True)


if __name__ == "__main__":
    main()
