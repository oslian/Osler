"""A/B test whether JEPA real-data input extraction is diluting signal.

The checkpoint and cohort are held fixed.  Only the current-state provenance
path changes:

* legacy: historical evaluator, including derived/default hidden values;
* measurement_pure: only observed values enter the state vector, while mask
  and age encode missingness as used during training.

This is a factual observational diagnostic, not a retraining or causal test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dka_world_model import load_checkpoint
from train_intervention_jepa import choose_device, evaluate_mimic_proxy


def _comparison(legacy, pure):
    targets = sorted(
        set(legacy.get("mae", {})) | set(pure.get("mae", {}))
    )
    rows = []
    for target in targets:
        legacy_row = legacy.get("mae", {}).get(target, {})
        pure_row = pure.get("mae", {}).get(target, {})
        persistence = legacy_row.get("persistence")
        rows.append({
            "target": target,
            "n_legacy": legacy_row.get("n"),
            "n_measurement_pure": pure_row.get("n"),
            "persistence_mae": persistence,
            "legacy_mae": legacy_row.get("jepa"),
            "measurement_pure_mae": pure_row.get("jepa"),
            "pure_minus_legacy": (
                pure_row["jepa"] - legacy_row["jepa"]
                if pure_row.get("jepa") is not None
                and legacy_row.get("jepa") is not None else None
            ),
            "pure_beats_persistence": (
                pure_row["jepa"] < persistence
                if pure_row.get("jepa") is not None
                and persistence is not None else None
            ),
            "legacy_beats_persistence": (
                legacy_row["jepa"] < persistence
                if legacy_row.get("jepa") is not None
                and persistence is not None else None
            ),
        })

    active_legacy = legacy.get("mae_active_dka", {})
    active_pure = pure.get("mae_active_dka", {})
    active_rows = []
    for target in sorted(set(active_legacy) | set(active_pure)):
        legacy_row = active_legacy.get(target, {})
        pure_row = active_pure.get(target, {})
        persistence = legacy_row.get("persistence")
        active_rows.append({
            "target": target,
            "n_legacy": legacy_row.get("n"),
            "n_measurement_pure": pure_row.get("n"),
            "persistence_mae": persistence,
            "legacy_mae": legacy_row.get("jepa"),
            "measurement_pure_mae": pure_row.get("jepa"),
            "pure_minus_legacy": (
                pure_row["jepa"] - legacy_row["jepa"]
                if pure_row.get("jepa") is not None
                and legacy_row.get("jepa") is not None else None
            ),
        })
    return {"all_window": rows, "active_dka": active_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument("--mimic", default="dka_transitions_6h_demo_v4.parquet")
    parser.add_argument(
        "--output", default="mimic_jepa_input_provenance_ablation.json"
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    legacy = evaluate_mimic_proxy(
        model, args.mimic, torch.device(device), input_mode="legacy"
    )
    measurement_pure = evaluate_mimic_proxy(
        model, args.mimic, torch.device(device), input_mode="measurement_pure"
    )
    report = {
        "test_type": "JEPA input provenance A/B ablation",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "cohort": str(Path(args.mimic).resolve()),
        "checkpoint_model": payload.get("metadata", {}).get("model"),
        "legacy": legacy,
        "measurement_pure": measurement_pure,
        "comparison": _comparison(legacy, measurement_pure),
        "interpretation_boundary": {
            "factual_only": True,
            "causal_claims_allowed": False,
            "retraining_performed": False,
            "checkpoint_promotion_allowed": False,
            "next_gate_if_pure_improves": (
                "rerun patient-heldout and hospital-heldout evaluation before retraining"
            ),
        },
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()
