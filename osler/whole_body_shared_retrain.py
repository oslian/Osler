"""Full-cohort retrain for the shared whole-body candidate.

The checkpoint is promoted only at target-horizon cells explicitly listed by
the independent gate.  The global shared latent remains candidate-only unless
the complete module-balanced gate passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from whole_body_shared_jepa import fit_shared as fit_shared_v1, load_shared_cohort


def _strict_cells(gate: dict) -> list[str]:
    if gate.get("validated_target_horizon_cells") is not None:
        return list(gate["validated_target_horizon_cells"])
    patient = gate["patient_heldout"]["aggregate"]
    conformal = gate["patient_heldout"]["conformal"]
    hospital = gate["hospital_heldout"]
    module_cells = {}
    for cell in gate["module_balanced"]["cells"]:
        key = f"{cell['target']}@{int(cell['horizon_hours']) if float(cell['horizon_hours']).is_integer() else cell['horizon_hours']}h"
        module_cells.setdefault(key, []).append(cell)
    output = []
    for key, value in patient.items():
        if not value.get("pass_all_splits"):
            continue
        if not conformal.get(key, {}).get("pass_all_splits"):
            continue
        if not hospital["values"].get(key, {}).get("pass_value"):
            continue
        if not hospital["conformal"].get(key, {}).get("pass"):
            continue
        cells = module_cells.get(key, [])
        if len(cells) >= 3 and sum(cell["pass"] for cell in cells) / len(cells) >= 0.80:
            output.append(key)
    return sorted(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--modules", required=True)
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--architecture", choices=("v1", "v2", "v3"), default="v1")
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    frame, variables, module_names = load_shared_cohort(args.data_root, modules, horizons, args.max_stays)
    rows = np.arange(len(frame), dtype=np.int64)
    if args.architecture == "v2":
        from whole_body_shared_jepa_v2 import fit_shared_v2 as fitter
    elif args.architecture == "v3":
        from whole_body_shared_jepa_v3 import fit_shared_v3 as fitter
    else:
        fitter = fit_shared_v1
    model, arrays, losses = fitter(
        frame, variables, len(module_names), rows, seed=2026,
        epochs=args.epochs, batch_size=2048,
    )
    validated_cells = _strict_cells(gate)
    checkpoint = {
        "model_type": type(model).__name__,
        "state_dict": model.state_dict(),
        "scaler": arrays[0].to_dict(),
        "variables": list(variables),
        "modules": list(module_names),
        "input_dim": model.input_dim,
        "hidden": model.hidden,
        "latent": model.latent,
        "promotion_status": "candidate_only",
        "architecture": {
            "shared_history_encoder": "GRU",
            "module_embedding": True,
            "architecture_version": args.architecture,
            "measurement_pure": True,
            "shared_latent_validated": False,
            "target_horizon_evidence_only": True,
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
        "gate_artifact": str(args.gate.resolve()),
        "validated_target_horizon_evidence": validated_cells,
        "training": {
            "rows": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "max_stays_per_horizon": args.max_stays,
            "loss_history": losses,
        },
    }
    torch.save(checkpoint, args.output)
    report = {
        "schema": f"whole_body_shared_jepa_full_retrain.{args.architecture}",
        "checkpoint": str(args.output.resolve()),
        "gate": str(args.gate.resolve()),
        "modules": list(module_names),
        "variables": list(variables),
        "rows": int(len(frame)),
        "loss_history": losses,
        "promotion_status": "candidate_only",
        "shared_latent_validated": False,
        "target_horizon_evidence_cells": validated_cells,
        "target_horizon_evidence_count": len(validated_cells),
        "gate_policy": {
            "global_shared_latent_requires_module_balanced_all_cells": True,
            "validated_cell_policy": "only target-horizon cells in the independent gate registry may move",
            "unvalidated_target_action": "per_module_runtime_registry_or_persistence",
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
