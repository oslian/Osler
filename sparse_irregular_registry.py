"""Build a fail-closed registry for sparse irregular-time factual models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


TIME_BIN_TO_RUNTIME_HORIZONS = {
    "0_6h": (3.0,),
    "6_24h": (12.0,),
    "24_72h": (48.0,),
}


def _load(value: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _key(target: str, horizon: float) -> str:
    rendered = str(int(horizon)) if float(horizon).is_integer() else str(horizon)
    return f"{target}@{rendered}h"


def build_sparse_registry(
    value_audit: Mapping[str, Any] | str | Path,
    conformal_audit: Mapping[str, Any] | str | Path,
    *,
    external_database_audit: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Combine independent sparse-target evidence without upgrading claims.

    A model is runtime-eligible only when patient, care-unit, forward-
    trajectory, conformal and external-database evidence all pass. Existing
    historical artifacts that predate a gate therefore remain fail-closed.
    """

    value = _load(value_audit)
    conformal = _load(conformal_audit)
    external = _load(external_database_audit)
    calibrated = {
        (str(item.get("target")), str(item.get("time_bin")), str(item.get("model")))
        for item in conformal.get("calibrated_cells", ())
    }
    external_cells = {
        (str(item.get("target")), str(item.get("time_bin")), str(item.get("model")))
        for item in external.get("validated_cells", ())
    }
    registry: dict[str, Any] = {}
    for report in value.get("results", ()):
        target = str(report.get("target"))
        time_bin = str(report.get("time_bin"))
        horizons = TIME_BIN_TO_RUNTIME_HORIZONS.get(time_bin, ())
        patient_splits = list(report.get("patient_splits") or ())
        care = report.get("care_unit_holdout") or {}
        forward = report.get("forward_trajectory_holdout") or {}
        for model in ("raw_event", "log_event"):
            evidence_key = (target, time_bin, model)
            requirements = {
                "patient_7_seed": bool(
                    len(patient_splits) >= 7
                    and all(
                        split.get(model, {}).get("significant", False)
                        for split in patient_splits
                    )
                ),
                "care_unit": bool(care.get(model, {}).get("significant", False)),
                "forward_trajectory": bool(
                    forward.get(model, {}).get("significant", False)
                ),
                "conformal": evidence_key in calibrated,
                "external_database": evidence_key in external_cells,
            }
            can_move = bool(all(requirements.values()))
            for horizon in horizons:
                cell_key = _key(target, horizon)
                candidate = {
                    "validated": can_move,
                    "validated_for_joint_runtime": False,
                    "irregular_time_forecast_validated": can_move,
                    "can_move": can_move,
                    "status": (
                        "validated_sparse_event_move"
                        if can_move
                        else "fallback_persistence"
                    ),
                    "source_model": f"sparse_irregular_{model}",
                    "time_bin": time_bin,
                    "runtime_query_horizon_hours": horizon,
                    "requirements": requirements,
                    "rejection_reasons": [
                        name for name, passed in requirements.items() if not passed
                    ],
                    "causal_claim_allowed": False,
                }
                previous = registry.get(cell_key)
                if previous is None or (
                    candidate["can_move"] and not previous["can_move"]
                ):
                    registry[cell_key] = candidate
    validated = sorted(key for key, cell in registry.items() if cell["can_move"])
    return {
        "schema": "sparse_irregular_promotion_registry.v1",
        "target_horizon_registry": registry,
        "validated_target_horizon_cells": validated,
        "validated_target_horizon_count": len(validated),
        "policy": {
            "source_specific_predictions_required": True,
            "requires_forward_trajectory_holdout": True,
            "requires_external_database_audit": True,
            "missing_evidence_action": "persistence_or_missing",
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
        },
        "promotion_status": (
            "validated_sparse_event_research_only"
            if validated
            else "candidate_only"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-audit", required=True, type=Path)
    parser.add_argument("--conformal-audit", required=True, type=Path)
    parser.add_argument("--external-database-audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_sparse_registry(
        args.value_audit,
        args.conformal_audit,
        external_database_audit=args.external_database_audit,
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "validated_cells": result["validated_target_horizon_count"],
                "promotion_status": result["promotion_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
