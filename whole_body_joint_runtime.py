"""Fail-closed runtime adapter for the joint whole-body JEPA.

The model may learn a shared body representation, but deployment is still
target-by-horizon gated.  This module deliberately accepts model predictions
as arrays rather than loading a checkpoint, so it can be used by both the
research evaluator and a future serving layer without changing training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


APPROVED_CHECKPOINT_STATUSES = {
    "validated",
    "promoted",
    "validated_research_only",
    "validated_target_gated_joint_research_only",
}
APPROVED_JOINT_REGISTRY_SCHEMA = "whole_body_promotion_registry.v3"
APPROVED_SPARSE_REGISTRY_SCHEMA = "sparse_irregular_promotion_registry.v1"
APPROVED_CANDIDATE_CATALOG_SCHEMA = "whole_body_runtime_candidate_catalog.v1"

IRREGULAR_TIME_TARGETS = {
    "inr",
    "ptt",
    "fibrinogen",
    "cortisol",
}

DIRECT_OBSERVATION_OR_NOWCAST_ONLY_TARGETS = {
    "troponin_i",
    "troponin_t",
    "tsh",
    "free_t4",
    "total_t4",
    "pth",
    "tpo_antibody",
    "afp",
    "cea",
    "ca_125",
    "ca_19_9",
    "psa",
}


def load_joint_registry(path: str | Path) -> dict[str, Any]:
    """Load a materialized promotion registry without upgrading raw gates."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "target_horizon_registry" in payload:
        return payload
    return {
        "schema": payload.get("schema"),
        "target_horizon_registry": {},
        "load_rejection_reason": "not_a_materialized_promotion_registry",
    }


def load_candidate_catalog(path: str | Path) -> dict[str, Any]:
    """Load the current materialized runtime catalog without upgrading it."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != APPROVED_CANDIDATE_CATALOG_SCHEMA:
        return {
            "schema": payload.get("schema"),
            "candidates": [],
            "load_rejection_reason": "runtime_candidate_catalog_v1_required",
        }
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        return {
            "schema": payload.get("schema"),
            "candidates": [],
            "load_rejection_reason": "catalog_candidates_must_be_a_list",
        }
    return payload


def resolve_catalog_cell(
    catalog: Mapping[str, Any] | str | Path,
    *,
    target: str,
    horizon: float,
    input_contract_id: str,
    variable_unit: str,
) -> dict[str, Any]:
    """Resolve exactly one executable source for a live target-horizon cell."""

    if isinstance(catalog, (str, Path)):
        catalog = load_candidate_catalog(catalog)
    key = _cell_key(target, horizon)
    reasons = []
    if catalog.get("schema") != APPROVED_CANDIDATE_CATALOG_SCHEMA:
        reasons.append("runtime_candidate_catalog_v1_required")
    matches = []
    for candidate in catalog.get("candidates", []):
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("runtime_status") != "target_gated_factual_research_only":
            continue
        if candidate.get("input_contract_id") != input_contract_id:
            continue
        registry = candidate.get("target_horizon_registry", {})
        if not isinstance(registry, Mapping) or key not in registry:
            continue
        if candidate.get("variable_units", {}).get(target) != variable_unit:
            continue
        matches.append(candidate)
    if not matches:
        reasons.append("no_exact_catalog_candidate")
    if len(matches) > 1:
        reasons.append("ambiguous_catalog_candidate")
    if reasons:
        return {
            "resolved": False,
            "target_horizon": key,
            "candidate_id": None,
            "registry": None,
            "rejection_reasons": reasons,
        }
    candidate = matches[0]
    return {
        "resolved": True,
        "target_horizon": key,
        "candidate_id": candidate.get("candidate_id"),
        "registry": {
            "schema": APPROVED_JOINT_REGISTRY_SCHEMA,
            "candidate_id": candidate.get("candidate_id"),
            "input_contract_id": candidate.get("input_contract_id"),
            "input_contract": candidate.get("input_contract"),
            "variable_units": candidate.get("variable_units", {}),
            "output_contract": candidate.get("output_contract", {}),
            "target_horizon_registry": {key: candidate["target_horizon_registry"][key]},
        },
        "rejection_reasons": [],
    }


def forecast_catalog_value(
    *,
    prediction: float,
    prediction_candidate_id: str,
    current: float,
    target: str,
    horizon: float,
    catalog: Mapping[str, Any] | str | Path,
    input_contract_id: str,
    variable_unit: str,
    checkpoint_status: str = "validated_target_gated_joint_research_only",
) -> dict[str, Any]:
    """Resolve and serve one catalog-backed factual prediction fail-closed."""

    resolution = resolve_catalog_cell(
        catalog,
        target=target,
        horizon=horizon,
        input_contract_id=input_contract_id,
        variable_unit=variable_unit,
    )
    if not resolution["resolved"]:
        return {
            "schema": "whole_body_catalog_forecast.v1",
            "target": target,
            "horizon_hours": float(horizon),
            "forecast_point": float(current) if np.isfinite(current) else None,
            "source_model": "persistence_fallback",
            "can_move": False,
            "fallback_reason": resolution["rejection_reasons"][0],
            "rejection_reasons": resolution["rejection_reasons"],
            "causal_claim_allowed": False,
        }
    if prediction_candidate_id != resolution["candidate_id"]:
        return {
            "schema": "whole_body_catalog_forecast.v1",
            "target": target,
            "horizon_hours": float(horizon),
            "candidate_id": resolution["candidate_id"],
            "forecast_point": float(current) if np.isfinite(current) else None,
            "source_model": "persistence_fallback",
            "can_move": False,
            "fallback_reason": "prediction_source_mismatch",
            "rejection_reasons": ["prediction_source_mismatch"],
            "causal_claim_allowed": False,
        }
    cell = forecast_joint_values(
        prediction=[[prediction]],
        current=[[current]],
        variables=(target,),
        horizons=horizon,
        registry=resolution["registry"],
        checkpoint_status=checkpoint_status,
        input_contract_id=input_contract_id,
        variable_units={target: variable_unit},
    )["forecasts"][0][target]
    return {
        "schema": "whole_body_catalog_forecast.v1",
        "target": target,
        "horizon_hours": float(horizon),
        "candidate_id": resolution["candidate_id"],
        **cell,
    }


def _cell_key(target: str, horizon: float) -> str:
    value = float(horizon)
    rendered = str(int(value)) if value.is_integer() else str(value)
    return f"{target}@{rendered}h"


def _registry_cell(registry: Mapping[str, Any], target: str, horizon: float) -> Mapping[str, Any]:
    cells = registry.get("target_horizon_registry", registry.get("registry", {}))
    if not isinstance(cells, Mapping):
        return {}
    value = cells.get(_cell_key(target, horizon), {})
    return value if isinstance(value, Mapping) else {}


def forecast_joint_values(
    prediction: np.ndarray,
    current: np.ndarray,
    variables: Sequence[str],
    horizons: float | Sequence[float],
    registry: Mapping[str, Any] | str | Path,
    *,
    checkpoint_status: str = "candidate_only",
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    observed: np.ndarray | None = None,
    nowcast: np.ndarray | None = None,
    nowcast_validated: np.ndarray | None = None,
    measurement_age_hours: np.ndarray | None = None,
    actual_horizons: np.ndarray | None = None,
    organ_latent_sources: Sequence[str] | None = None,
    observed_treatment_context: Sequence[Mapping[str, Any]] | None = None,
    observation_probability: np.ndarray | None = None,
    stable_probability: np.ndarray | None = None,
    next_measurement_time_hours: np.ndarray | None = None,
    input_contract_id: str | None = None,
    variable_units: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply target-horizon promotion and persistence fallback.

    ``prediction`` and ``current`` are raw clinical-unit arrays with shape
    ``(rows, targets)`` or ``(targets,)``.  A prediction is allowed to move
    only when both the checkpoint status and its registry cell are validated.
    Every other value is returned unchanged from ``current``.
    """

    prediction = np.asarray(prediction, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if prediction.ndim == 1:
        prediction = prediction[None, :]
    if current.ndim == 1:
        current = current[None, :]
    if prediction.shape != current.shape:
        raise ValueError("prediction and current must have the same shape")
    if prediction.shape[1] != len(variables):
        raise ValueError("prediction width must equal variables length")
    if np.isscalar(horizons):
        horizon_values = np.full(prediction.shape[0], float(horizons), dtype=np.float64)
    else:
        horizon_values = np.asarray(horizons, dtype=np.float64).reshape(-1)
        if len(horizon_values) != prediction.shape[0]:
            raise ValueError("horizons length must equal prediction row count")
    if isinstance(registry, (str, Path)):
        registry = load_joint_registry(registry)
    registry_schema_valid = (
        registry.get("schema") == APPROVED_JOINT_REGISTRY_SCHEMA
    )
    expected_contract_id = registry.get("input_contract_id")
    input_contract_declared = bool(expected_contract_id)
    input_contract_valid = bool(
        input_contract_declared and input_contract_id == expected_contract_id
    )
    expected_units = registry.get("variable_units", {})
    supplied_units = variable_units or {}

    def optional_matrix(value, name, dtype=np.float64):
        if value is None:
            return None
        matrix = np.asarray(value, dtype=dtype)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape != prediction.shape:
            raise ValueError(f"{name} must have the same shape as prediction")
        return matrix

    lower = optional_matrix(lower, "lower")
    upper = optional_matrix(upper, "upper")
    observed = optional_matrix(observed, "observed", dtype=bool)
    nowcast = optional_matrix(nowcast, "nowcast")
    nowcast_validated = optional_matrix(
        nowcast_validated, "nowcast_validated", dtype=bool
    )
    measurement_age_hours = optional_matrix(
        measurement_age_hours, "measurement_age_hours"
    )
    actual_horizons = optional_matrix(actual_horizons, "actual_horizons")
    observation_probability = optional_matrix(
        observation_probability, "observation_probability"
    )
    stable_probability = optional_matrix(stable_probability, "stable_probability")
    next_measurement_time_hours = optional_matrix(
        next_measurement_time_hours, "next_measurement_time_hours"
    )
    if organ_latent_sources is not None and len(organ_latent_sources) != len(variables):
        raise ValueError("organ_latent_sources length must equal variables length")

    approved_checkpoint = checkpoint_status in APPROVED_CHECKPOINT_STATUSES
    anchor_values = current.copy()
    if observed is not None:
        anchor_values[~observed] = np.nan
    if nowcast is not None and nowcast_validated is not None:
        use_nowcast = (
            ~np.isfinite(anchor_values)
            & nowcast_validated
            & np.isfinite(nowcast)
        )
        anchor_values[use_nowcast] = nowcast[use_nowcast]
    output = anchor_values.copy()
    forecasts = []
    for row_index, horizon in enumerate(horizon_values):
        row_forecast = {}
        for target_index, target in enumerate(variables):
            cell = _registry_cell(registry, target, horizon)
            cell_validated = bool(
                cell.get(
                    "validated_for_joint_runtime",
                    cell.get("can_move", cell.get("validated", False)),
                )
            )
            reasons = []
            if not approved_checkpoint:
                reasons.append("checkpoint_candidate_only")
            if not registry_schema_valid:
                reasons.append("joint_promotion_registry_v2_required")
            if not input_contract_declared:
                reasons.append("registry_input_contract_missing")
            if not input_contract_valid:
                reasons.append("input_contract_mismatch")
            expected_unit = expected_units.get(target)
            unit_contract_declared = expected_unit is not None
            if not unit_contract_declared:
                reasons.append("registry_variable_unit_missing")
            if not unit_contract_declared or supplied_units.get(target) != expected_unit:
                reasons.append("variable_unit_mismatch")
            if not cell_validated:
                reasons.append("target_horizon_not_validated")
            if (
                target in IRREGULAR_TIME_TARGETS
                and not bool(cell.get("irregular_time_forecast_validated", False))
            ):
                reasons.append("requires_irregular_time_event_validation")
            if (
                target in DIRECT_OBSERVATION_OR_NOWCAST_ONLY_TARGETS
                and not bool(cell.get("repeated_event_support_validated", False))
            ):
                reasons.append("future_event_support_not_validated")
            candidate = prediction[row_index, target_index]
            observed_anchor = current[row_index, target_index]
            anchor = anchor_values[row_index, target_index]
            sparse_policy_pass = not any(
                reason
                in {
                    "requires_irregular_time_event_validation",
                    "future_event_support_not_validated",
                }
                for reason in reasons
            )
            can_move = (
                approved_checkpoint
                and registry_schema_valid
                and input_contract_valid
                and unit_contract_declared
                and cell_validated
                and sparse_policy_pass
                and "variable_unit_mismatch" not in reasons
                and np.isfinite(candidate)
                and np.isfinite(anchor)
            )
            if can_move:
                output[row_index, target_index] = candidate
            else:
                output[row_index, target_index] = anchor
            anchor_is_observed = bool(
                observed[row_index, target_index]
                if observed is not None
                else np.isfinite(observed_anchor)
            )
            nowcast_is_validated = bool(
                nowcast is not None
                and nowcast_validated is not None
                and nowcast_validated[row_index, target_index]
                and np.isfinite(nowcast[row_index, target_index])
            )
            if can_move:
                source = str(cell.get("source_model", "joint_jepa_validated"))
                status = "validated_move"
            elif np.isfinite(anchor):
                source = "persistence_fallback"
                status = "fallback_persistence"
            else:
                source = "missing_unresolved"
                status = "missing"
                reasons.append("no_observed_or_validated_nowcast_anchor")
            interval_ready = bool(
                can_move
                and lower is not None
                and upper is not None
                and cell.get("patient_conformal", False)
                and cell.get("hospital_conformal", False)
                and cell.get("care_unit_conformal", False)
                and cell.get("time_conformal", False)
            )
            entry = {
                "task_type": str(
                    cell.get(
                        "task_type",
                        registry.get("output_contract", {})
                        .get(target, {})
                        .get("task_type", "continuous_value"),
                    )
                ),
                "observed_value": (
                    float(observed_anchor)
                    if anchor_is_observed and np.isfinite(observed_anchor)
                    else None
                ),
                "nowcast": (
                    float(nowcast[row_index, target_index])
                    if nowcast_is_validated and not anchor_is_observed
                    else None
                ),
                "forecast_point": float(output[row_index, target_index]) if np.isfinite(output[row_index, target_index]) else None,
                # v1 compatibility field.
                "point": float(output[row_index, target_index]) if np.isfinite(output[row_index, target_index]) else None,
                "source_model": source,
                "source": source,
                "organ_latent_source": (
                    str(organ_latent_sources[target_index])
                    if organ_latent_sources is not None
                    else "shared_body_plus_target_specific"
                ),
                "observed_treatment_context": (
                    dict(observed_treatment_context[row_index])
                    if observed_treatment_context is not None
                    and row_index < len(observed_treatment_context)
                    else {}
                ),
                "can_move": bool(can_move),
                "status": status,
                "rejection_reasons": reasons,
                "fallback_reason": reasons[0] if reasons else None,
                "horizon_hours": float(horizon),
                "actual_horizon": float(
                    actual_horizons[row_index, target_index]
                    if actual_horizons is not None
                    and np.isfinite(actual_horizons[row_index, target_index])
                    else horizon
                ),
                "measurement_age": float(
                    measurement_age_hours[row_index, target_index]
                )
                if measurement_age_hours is not None
                and np.isfinite(measurement_age_hours[row_index, target_index])
                else None,
                "causal_claim_allowed": False,
            }
            if interval_ready:
                lower_value = lower[row_index, target_index]
                upper_value = upper[row_index, target_index]
                if np.isfinite(lower_value) and np.isfinite(upper_value):
                    entry["lower"] = float(lower_value)
                    entry["upper"] = float(upper_value)
                    entry["interval_status"] = "calibrated"
            if "interval_status" not in entry:
                entry["lower"] = None
                entry["upper"] = None
                entry["interval_status"] = "needs_calibration_audit"
            entry["current_estimate_source"] = (
                "observed"
                if anchor_is_observed and np.isfinite(observed_anchor)
                else "validated_nowcast"
                if nowcast_is_validated
                else "missing"
            )
            if observation_probability is not None:
                value = observation_probability[row_index, target_index]
                entry["observation_probability"] = (
                    float(value) if np.isfinite(value) else None
                )
            if stable_probability is not None:
                value = stable_probability[row_index, target_index]
                entry["stable_probability"] = (
                    float(value) if np.isfinite(value) else None
                )
            if next_measurement_time_hours is not None:
                value = next_measurement_time_hours[row_index, target_index]
                entry["next_measurement_time_hours"] = (
                    float(value) if np.isfinite(value) else None
                )
            row_forecast[target] = entry
        forecasts.append(row_forecast)
    return {
        "schema": "whole_body_state_forecast.v2",
        "forecasts": forecasts,
        "prediction_raw": output.tolist(),
        "semantics": {
            "shared_body_state": True,
            "organ_latent_tokens": True,
            "target_specific_heads": True,
            "sparse_cross_system_coupling": True,
            "measurement_process_separate": True,
            "persistence_fallback": True,
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
            "input_contract_id": input_contract_id,
        },
    }


def forecast_sparse_irregular_value(
    *,
    target: str,
    query_horizon_hours: float,
    current: float | None,
    prediction: float | None,
    actual_horizon_hours: float | None,
    registry: Mapping[str, Any] | str | Path,
    checkpoint_status: str = "candidate_only",
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, Any]:
    """Serve one source-specific irregular-event prediction fail-closed."""

    if isinstance(registry, (str, Path)):
        registry = load_joint_registry(registry)
    schema_valid = registry.get("schema") == APPROVED_SPARSE_REGISTRY_SCHEMA
    cell = _registry_cell(registry, target, query_horizon_hours)
    source = str(cell.get("source_model", ""))
    reasons = []
    if checkpoint_status != "validated_sparse_event_research_only":
        reasons.append("sparse_checkpoint_candidate_only")
    if not schema_valid:
        reasons.append("sparse_irregular_registry_v1_required")
    if not cell.get("irregular_time_forecast_validated", False):
        reasons.append("irregular_time_cell_not_validated")
    if not source.startswith("sparse_irregular_"):
        reasons.append("source_specific_sparse_model_required")
    anchor = float(current) if current is not None and np.isfinite(current) else None
    candidate = (
        float(prediction)
        if prediction is not None and np.isfinite(prediction)
        else None
    )
    can_move = bool(not reasons and anchor is not None and candidate is not None)
    point = candidate if can_move else anchor
    interval_ready = bool(
        can_move
        and lower is not None
        and upper is not None
        and np.isfinite(lower)
        and np.isfinite(upper)
        and cell.get("requirements", {}).get("conformal", False)
    )
    return {
        "schema": "whole_body_sparse_event_forecast.v1",
        "target": target,
        "observed_value": anchor,
        "forecast_point": point,
        "lower": float(lower) if interval_ready else None,
        "upper": float(upper) if interval_ready else None,
        "query_horizon_hours": float(query_horizon_hours),
        "actual_horizon": (
            float(actual_horizon_hours)
            if actual_horizon_hours is not None
            and np.isfinite(actual_horizon_hours)
            else None
        ),
        "source_model": source if can_move else "persistence_fallback",
        "can_move": can_move,
        "fallback_reason": reasons[0] if reasons else None,
        "rejection_reasons": reasons,
        "interval_status": "calibrated" if interval_ready else "needs_calibration_audit",
        "causal_claim_allowed": False,
    }


__all__ = [
    "forecast_catalog_value",
    "forecast_joint_values",
    "forecast_sparse_irregular_value",
    "load_candidate_catalog",
    "load_joint_registry",
    "resolve_catalog_cell",
]
