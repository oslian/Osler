"""Run the observation-only nonlinear rollout screen across body-system cohorts."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from audit_nonlinear_latent_rollout import run


DEFAULT_MODULES = (
    "acute_neuro",
    "aki",
    "cardiac_injury",
    "cardiovascular_instability",
    "coagulopathy_heme",
    "electrolyte_acid_base",
    "endocrine_stress",
    "gi_pancreatic_nutrition",
    "hepatic_failure",
    "immune_inflammatory",
    "integumentary_skin_wound",
    "musculoskeletal_rhabdo",
    "respiratory",
    "sepsis",
    "toxic_metabolic",
)


def available_horizons(data_root: Path, module: str) -> tuple[int, ...]:
    values = []
    for candidate in data_root.glob(f"eicu_{module}_transitions_*h.parquet"):
        try:
            values.append(int(candidate.stem.rsplit("_", 1)[-1][:-1]))
        except ValueError:
            continue
    return tuple(sorted(set(values)))


def summarize(report: dict[str, object]) -> dict[str, object]:
    patient = report["patient_heldout"]
    horizons = {}
    for horizon in report["horizons"]:
        deltas = []
        for split in patient.values():
            for cell in split.get("metrics", {}).get("by_horizon", {}).get(str(horizon), {}).values():
                value = cell.get("candidate_delta_vs_persistence")
                if value is not None and np.isfinite(value):
                    deltas.append(float(value))
        horizons[str(horizon)] = {
            "median_delta_vs_persistence": float(np.median(deltas)) if deltas else None,
            "cells": int(len(deltas)),
            "negative_cells": int(sum(value < 0 for value in deltas)),
        }
    stability = [split["rollout_stability"] for split in patient.values()]
    return {
        "module": report["module"],
        "horizons": report["horizons"],
        "rows": report["rows"],
        "subjects": report["subjects"],
        "leakage_passed": report["leakage_audit"]["passed"],
        "horizon_summary": horizons,
        "patient_stability_bounded": f"{sum(item['bounded'] for item in stability)}/{len(stability)}",
        "hospital_stability_bounded": (
            report["hospital_heldout"]["rollout_stability"]["bounded"]
            if report.get("hospital_heldout") else None
        ),
        "status": "candidate_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-stays-per-horizon", type=int, default=800)
    parser.add_argument("--cap-train-rows-per-horizon", type=int, default=4000)
    parser.add_argument("--output", default="/private/tmp/nonlinear_latent_rollout_body_system_screen.json")
    args = parser.parse_args()
    root = Path(args.data_root)
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    summary = []
    failures = []
    for module in modules:
        horizons = available_horizons(root, module)
        if not horizons:
            failures.append({"module": module, "error": "no horizon cohorts"})
            continue
        output = Path(f"/private/tmp/nonlinear_latent_rollout_{module}_screen.json")
        try:
            report = run(
                root,
                module,
                horizons,
                seeds=args.seeds,
                history=args.history,
                hidden=args.hidden,
                latent=args.latent,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_stays_per_horizon=args.max_stays_per_horizon,
                cap_train_rows_per_horizon=args.cap_train_rows_per_horizon,
                output=output,
            )
            summary.append(summarize(report))
        except Exception as exc:
            failures.append({"module": module, "error": repr(exc)})
        finally:
            gc.collect()
    result = {
        "observation_only": True,
        "causal_claim_allowed": False,
        "counterfactual_claim_allowed": False,
        "clinical_claim_allowed": False,
        "status": "screening_candidate_only",
        "modules": summary,
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}; modules={len(summary)} failures={len(failures)}")


if __name__ == "__main__":
    main()
