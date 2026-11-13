"""One patient-state inference path with validated per-cell frozen readouts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from io import StringIO
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from nonlinear_latent_rollout import StateScaler
from osler_jepa.shared_release_features import build_target_head_features
from whole_body_joint_jepa import JointWholeBodyJEPAWorldModel

from .contracts import ALIASES, TIME_COLUMNS, VARIABLE_BOUNDS, VARIABLE_UNITS
from .nowcast import derived_current_completions, latest_nowcast_row
from .source_selection import HEAD, MULTI, load_selector, predict_head_result


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models"
DEFAULT_HORIZONS = (1, 3, 6, 12, 24, 48)
HISTORY_LAGS_HOURS = (48.0, 24.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.0)


@dataclass(frozen=True)
class ReleaseAssets:
    model: JointWholeBodyJEPAWorldModel
    scaler: StateScaler
    checkpoint: dict[str, Any]
    heads: dict[str, Any]
    registry: dict[str, Any]
    intervals: dict[str, Any]
    event_heads: dict[str, Any]
    nowcast: dict[str, Any]
    source_selector: dict[str, Any]
    multisource_heads: dict[str, Any]
    max_age_hours: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _column_name(value: object) -> str:
    return (
        str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    )


def _relative_hours(frame: pd.DataFrame) -> pd.Series:
    for name in TIME_COLUMNS:
        if name not in frame:
            continue
        numeric = pd.to_numeric(frame[name], errors="coerce")
        if numeric.notna().all():
            return numeric.astype(float) - float(numeric.min())
        timestamps = pd.to_datetime(frame[name], errors="coerce")
        if timestamps.notna().all():
            return (timestamps - timestamps.min()).dt.total_seconds() / 3600.0
    if len(frame) == 1:
        return pd.Series([0.0], index=frame.index)
    raise ValueError("multiple rows require hours_since_onset or a timestamp column")


def parse_patient_csv(text: str, variables: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Parse wide or variable/value CSV; blank is missing and numeric zero is observed."""

    if not text.strip():
        raise ValueError("patient CSV is empty")
    raw = pd.read_csv(StringIO(text))
    if raw.empty:
        raise ValueError("patient CSV has no data rows")
    raw.columns = [_column_name(column) for column in raw.columns]
    if {"variable", "value"}.issubset(raw.columns):
        raw["variable"] = raw["variable"].map(_column_name).replace(ALIASES)
        raw["hours_since_onset"] = _relative_hours(raw)
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw = (
            raw.pivot_table(
                index="hours_since_onset",
                columns="variable",
                values="value",
                aggfunc="last",
            )
            .reset_index()
            .rename_axis(columns=None)
        )
    else:
        raw.rename(columns={name: ALIASES.get(name, name) for name in raw.columns}, inplace=True)
        raw["hours_since_onset"] = _relative_hours(raw)

    accepted = set(variables or VARIABLE_UNITS) | {"sbp", "dbp"}
    present = [column for column in raw.columns if column in accepted]
    if not present:
        raise ValueError("patient CSV contains no recognized physiological columns")
    frame = raw[["hours_since_onset", *present]].copy()
    for column in present:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "map" not in frame:
        frame["map"] = np.nan
    if {"sbp", "dbp"}.issubset(frame.columns):
        derived = (frame["sbp"] + 2.0 * frame["dbp"]) / 3.0
        frame["map"] = frame["map"].fillna(derived)
    frame = frame.sort_values("hours_since_onset", kind="stable")
    frame = frame.groupby("hours_since_onset", as_index=False, sort=True).last()
    physiological = [column for column in frame.columns if column != "hours_since_onset"]
    frame = frame.loc[frame[physiological].notna().any(axis=1)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("patient CSV has no finite physiological values")
    return frame


def _checkpoint_scaler(checkpoint: dict[str, Any]) -> StateScaler:
    payload = checkpoint["scaler"]
    return StateScaler(
        tuple(payload["variables"]),
        np.asarray(payload["medians"], dtype=np.float32),
        np.asarray(payload["scales"], dtype=np.float32),
    )


@lru_cache(maxsize=4)
def load_release(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    require_event_heads: bool = True,
) -> ReleaseAssets:
    """Load and strictly validate model, head bundle, registry, and feature contract."""

    model_dir = Path(model_dir).resolve()
    checkpoint = torch.load(
        model_dir / "whole_body_joint_jepa.pt",
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["state_dict"]
    architecture = checkpoint["architecture"]
    model = JointWholeBodyJEPAWorldModel(
        n_state=len(checkpoint["variables"]),
        input_dim=int(checkpoint["input_dim"]),
        module_count=len(checkpoint["modules"]),
        hidden=int(checkpoint["hidden"]),
        latent=int(checkpoint["latent"]),
        token_dim=int(state["variable_embedding"].shape[1]),
        variable_module_ids=state["variable_module_ids"].numpy(),
        variable_module_membership=state["variable_module_membership"].numpy(),
        horizon_values=state["horizon_bucket_values"].numpy(),
        regime_count=int(architecture["regime_count"]),
        target_horizon_regime_adapter=bool(architecture["target_horizon_regime_adapter"]),
        patient_residual_adapter=bool(architecture["patient_residual_adapter"]),
        uncertainty_gate=bool(architecture["uncertainty_gate"]),
        validated_edge_adapters=bool(architecture["validated_edge_adapters"]),
        validated_edge_specs=(),
        measurement_time_head=bool(architecture["next_measurement_time_head"]),
        belief_input_mode="observed",
        target_momentum=float(architecture["target_momentum"]),
        target_min_observations=int(architecture["target_min_observations"]),
        target_encoder_mode=str(architecture["target_encoder_mode"]),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    heads = joblib.load(model_dir / "unified_shared_jepa_head_bundle.joblib")
    if heads.get("schema") != "unified_shared_jepa_head_bundle.v1":
        raise ValueError("unsupported target-head bundle")
    registry = json.loads((model_dir / "runtime_registry.json").read_text(encoding="utf-8"))
    intervals = json.loads(
        (model_dir / "interval_calibrator.json").read_text(encoding="utf-8")
    )
    if intervals.get("schema") != "whole_body_jepa.interval_calibrator.v1":
        raise ValueError("unsupported interval calibrator")
    for name, expected in intervals["release_binding"].items():
        if _sha256(model_dir / name) != expected:
            raise ValueError(f"interval calibrator release mismatch: {name}")
    event_heads: dict[str, Any] = {"validated_heads": [], "heads": {}}
    if require_event_heads:
        event_heads = joblib.load(model_dir / "event_risk_heads.joblib")
        if event_heads.get("schema") != "whole_body_jepa.event_risk_heads.v1":
            raise ValueError("unsupported event-risk head bundle")
        if int(event_heads["feature_width"]) != 311:
            raise ValueError("event-risk feature width must be 311")
        for name, expected in event_heads["release_binding"].items():
            if _sha256(model_dir / name) != expected:
                raise ValueError(f"event-risk head release mismatch: {name}")
    contract = json.loads((model_dir / "FEATURE_CONTRACT.json").read_text(encoding="utf-8"))
    if int(contract["feature_width"]) != 311:
        raise ValueError("release feature width must be 311")
    for cell, head in heads["heads"].items():
        if int(head["feature_width"]) != 311:
            raise ValueError(f"{cell} has incompatible feature width")
    nowcast = joblib.load(model_dir / "cross_target_nowcast.joblib")
    if nowcast.get("schema") != "whole_body_jepa.cross_target_nowcast.v1":
        raise ValueError("unsupported cross-target nowcast bundle")
    for name, expected in nowcast["release_binding"].items():
        if _sha256(model_dir / name) != expected:
            raise ValueError(f"cross-target nowcast release mismatch: {name}")
    selector = load_selector(model_dir / "unified_source_selector.json", model_dir)
    if set(selector["cells"]) != {cell["cell"] for cell in registry["cells"]}:
        raise ValueError("source selector does not cover the release catalog")
    multisource = joblib.load(model_dir / "multisource_shared_jepa_head_bundle.joblib")
    if multisource.get("schema") != "multisource_shared_jepa_head_bundle.v1":
        raise ValueError("unsupported multi-source head bundle")
    for cell, decision in selector["cells"].items():
        if decision["source_model"] == HEAD and cell not in heads["heads"]:
            raise ValueError(f"selected shared head missing: {cell}")
        if decision["source_model"] == MULTI and cell not in multisource["heads"]:
            raise ValueError(f"selected multi-source head missing: {cell}")
    return ReleaseAssets(
        model=model,
        scaler=_checkpoint_scaler(checkpoint),
        checkpoint=checkpoint,
        heads=heads,
        registry=registry,
        intervals=intervals,
        event_heads=event_heads,
        nowcast=nowcast,
        source_selector=selector,
        multisource_heads=multisource,
        max_age_hours=float(contract["asof_contract"]["release_max_age_hours"]),
    )


def _model_frame(
    frame: pd.DataFrame,
    variables: tuple[str, ...],
    max_age_hours: float,
    patient_id: str,
) -> pd.DataFrame:
    hours = frame["hours_since_onset"].to_numpy(dtype=np.float64)
    columns: dict[str, object] = {
        "subject_id": patient_id,
        "stay_id": patient_id,
        "hours_since_onset": hours,
        "_time_hours": hours,
    }
    for variable in variables:
        observed = (
            pd.to_numeric(frame[variable], errors="coerce")
            if variable in frame
            else pd.Series(np.nan, index=frame.index)
        )
        carried = observed.ffill()
        last_time = pd.Series(np.where(observed.notna(), hours, np.nan), index=frame.index).ffill()
        age = hours - last_time.to_numpy(dtype=np.float64)
        carried = carried.where(np.isfinite(age) & (age <= max_age_hours))
        columns[f"{variable}_t"] = carried.to_numpy(dtype=np.float64)
        columns[f"{variable}_age_hr"] = age
    return pd.DataFrame(columns)


def _multiscale_history_indices(frame: pd.DataFrame) -> np.ndarray:
    lags = HISTORY_LAGS_HOURS
    output = np.zeros((len(frame), len(lags)), dtype=np.int64)
    times = frame["_time_hours"].to_numpy(dtype=np.float64)
    for positions in frame.groupby("stay_id", sort=False).indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        positions = positions[np.argsort(times[positions], kind="stable")]
        ordered = times[positions]
        for local_index, row in enumerate(positions):
            available_positions = positions[: local_index + 1]
            available_times = ordered[: local_index + 1]
            anchor = ordered[local_index]
            for lag_index, lag in enumerate(lags):
                prior = np.flatnonzero(available_times <= anchor - lag + 1e-9)
                output[row, lag_index] = available_positions[prior[-1] if len(prior) else 0]
    return output


def _build_inputs(frame: pd.DataFrame, assets: ReleaseAssets, horizon: int) -> dict[str, np.ndarray]:
    current, _, _ = assets.scaler.normalize_current(frame)
    mask = np.ones_like(current, dtype=np.float32)
    ages = np.zeros_like(current, dtype=np.float32)
    presence = np.ones((len(frame), len(assets.checkpoint["modules"])), dtype=np.float32)
    onset = np.clip(frame["hours_since_onset"].to_numpy(dtype=np.float32), 0.0, 168.0)[:, None] / 24.0
    elapsed = np.clip(frame["_time_hours"].to_numpy(dtype=np.float32), 0.0, 168.0)[:, None] / 168.0
    matrix = np.concatenate([current, mask, ages, onset, presence, elapsed], axis=1).astype(np.float32)
    if matrix.shape[1] != int(assets.checkpoint["input_dim"]):
        raise ValueError("runtime input width does not match checkpoint")
    return {
        "input": matrix,
        "current": current,
        "mask": mask,
        "ages": ages,
        "presence": presence,
        "history": _multiscale_history_indices(frame),
        "horizon": np.full(len(frame), float(horizon), dtype=np.float32),
    }


def _predict_head(
    head: dict[str, Any], features: np.ndarray, current: float, *, runtime: bool
) -> tuple[float, bool]:
    values, abstained = predict_head_result(
        head, features, np.asarray([current]), runtime=runtime
    )
    return float(values[0]), bool(abstained[0])


def _predict_interval_scale(model: dict[str, Any], features: np.ndarray) -> float:
    matrix = np.asarray(features, dtype=np.float64)
    center = np.asarray(model["center"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    beta = np.asarray(model["beta"], dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, center)
    standardized = np.clip((matrix - center) / scale, -8.0, 8.0)
    design = np.concatenate(
        [np.ones((len(standardized), 1)), standardized], axis=1
    )
    prediction = np.exp(np.clip(design @ beta, -20.0, 20.0))
    return float(np.clip(prediction[0], model["lower"], model["upper"]))


def _predict_event_risk(model: Any, features: np.ndarray) -> float:
    probability = float(model.predict_proba(np.asarray(features, dtype=np.float64))[0, 1])
    return float(np.clip(probability, 0.0, 1.0))


def _current_state(frame: pd.DataFrame, variables: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    hours = frame["hours_since_onset"].to_numpy(dtype=np.float64)
    output: dict[str, dict[str, Any]] = {}
    for variable in variables:
        values = (
            pd.to_numeric(frame[variable], errors="coerce").to_numpy(dtype=np.float64)
            if variable in frame
            else np.full(len(frame), np.nan)
        )
        observed = np.flatnonzero(np.isfinite(values))
        if not len(observed):
            output[variable] = {"observed": False, "value": None, "age_hours": None, "count": 0}
            continue
        latest = int(observed[-1])
        output[variable] = {
            "observed": True,
            "value": float(values[latest]),
            "age_hours": float(hours[-1] - hours[latest]),
            "count": int(len(observed)),
        }
    return output


def _completed_current_state(
    state: dict[str, dict[str, Any]],
    model_frame: pd.DataFrame,
    assets: ReleaseAssets,
) -> dict[str, dict[str, Any]]:
    fresh_values = {
        variable: payload["value"]
        for variable, payload in state.items()
        if payload["observed"]
        and payload["age_hours"] is not None
        and float(payload["age_hours"]) <= assets.max_age_hours
    }
    derived = derived_current_completions(fresh_values)
    completed: dict[str, dict[str, Any]] = {}
    for variable in assets.checkpoint["variables"]:
        payload = state[variable]
        if payload["observed"]:
            completed[variable] = {
                "value": payload["value"],
                "source": "observed",
                "lower": None,
                "upper": None,
                "interval_status": "not_applicable_observed",
                "inputs_observed": None,
            }
            continue
        if variable in derived:
            value = float(derived[variable]["value"])
            completed[variable] = {
                "value": value,
                "source": "derived_formula",
                "lower": value,
                "upper": value,
                "interval_status": "deterministic_formula",
                "inputs_observed": derived[variable]["inputs"],
            }
            continue
        head = assets.nowcast["heads"].get(variable)
        if head is None:
            completed[variable] = {
                "value": None,
                "source": "missing",
                "lower": None,
                "upper": None,
                "interval_status": "not_validated",
                "inputs_observed": 0,
            }
            continue
        feature_row = latest_nowcast_row(model_frame, head["feature_columns"])
        observed_count = int(np.isfinite(feature_row.to_numpy(dtype=np.float64)).sum())
        if observed_count < int(head["minimum_observed_features"]):
            completed[variable] = {
                "value": None,
                "source": "missing",
                "lower": None,
                "upper": None,
                "interval_status": "insufficient_related_observations",
                "inputs_observed": observed_count,
            }
            continue
        value = float(head["model"].predict(feature_row)[0])
        lower_bound = head.get("lower_bound")
        upper_bound = head.get("upper_bound")
        if lower_bound is not None:
            value = max(float(lower_bound), value)
        if upper_bound is not None:
            value = min(float(upper_bound), value)
        radius = head.get("interval_radius")
        lower = None if radius is None else value - float(radius)
        upper = None if radius is None else value + float(radius)
        if lower is not None and lower_bound is not None:
            lower = max(float(lower_bound), lower)
        if upper is not None and upper_bound is not None:
            upper = min(float(upper_bound), upper)
        completed[variable] = {
            "value": value,
            "source": "validated_cross_target_nowcast",
            "lower": lower,
            "upper": upper,
            "interval_status": (
                "mimic_patient_split_conformal_90"
                if radius is not None
                else "point_only"
            ),
            "inputs_observed": observed_count,
        }
    return completed


def _hypoxemia(value: float | None) -> float | None:
    return None if value is None else float(value <= 92.0)


def predict_patient(
    csv_text: str,
    *,
    filename: str = "patient.csv",
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Condition every release cell on one patient's causal measurement history."""

    assets = load_release(model_dir)
    variables = tuple(assets.checkpoint["variables"])
    frame = parse_patient_csv(csv_text, variables)
    digest = hashlib.sha256(frame.to_csv(index=False, na_rep="").encode()).hexdigest()[:16]
    model_frame = _model_frame(frame, variables, assets.max_age_hours, digest)
    state = _current_state(frame, variables)
    completed_state = _completed_current_state(state, model_frame, assets)
    normalized_by_horizon: dict[int, np.ndarray] = {}
    physical_by_horizon: dict[int, np.ndarray] = {}
    outputs_by_horizon: dict[int, dict[str, torch.Tensor]] = {}
    current_by_horizon: dict[int, torch.Tensor] = {}

    for horizon in sorted({int(cell["horizon_hours"]) for cell in assets.registry["cells"]}):
        arrays = _build_inputs(model_frame, assets, horizon)
        row = len(model_frame) - 1
        current_tensor = torch.from_numpy(arrays["current"][[row]]).float()
        with torch.no_grad():
            output = assets.model(
                torch.from_numpy(arrays["input"][arrays["history"][[row]]]).float(),
                current_tensor,
                torch.from_numpy(arrays["mask"][[row]]).float(),
                torch.from_numpy(arrays["ages"][[row]]).float(),
                torch.from_numpy(arrays["presence"][[row]]).float(),
                torch.from_numpy(arrays["horizon"][[row]]).float(),
            )
        normalized = output["value"][0].cpu().numpy().astype(np.float64)
        physical = normalized * assets.scaler.scales + assets.scaler.medians
        normalized_by_horizon[horizon] = normalized
        physical_by_horizon[horizon] = physical
        outputs_by_horizon[horizon] = output
        current_by_horizon[horizon] = current_tensor

    validated_event_heads = set(assets.event_heads["validated_heads"])
    cells = []
    for contract in assets.registry["cells"]:
        cell = str(contract["cell"])
        target = str(contract["target"])
        horizon = int(contract["horizon_hours"])
        model_target = "o2sat" if target == "hypoxemia_event" else target
        target_state = state.get(model_target, {"observed": False, "value": None, "age_hours": None, "count": 0})
        current = target_state["value"]
        completion = completed_state.get(model_target, {})
        index = variables.index(model_target)
        candidate = float(physical_by_horizon[horizon][index])
        decision = assets.source_selector["cells"][cell]
        source = decision["source_model"]
        interval = assets.intervals["cells"].get(cell)
        is_event = target == "hypoxemia_event"
        head_abstained = False
        needs_feature = source in {HEAD, MULTI} or cell in validated_event_heads or bool(
            interval
            and interval.get("runtime_interval_method") == "normalized_conformal"
        )

        feature = None
        if needs_feature:
            output = outputs_by_horizon[horizon]
            current_tensor = current_by_horizon[horizon]
            module_index = int(assets.model.variable_module_ids[index].item())
            feature = build_target_head_features(
                shared_context=output["context_latent"][0].cpu().numpy(),
                patient_belief=output["belief_state"][0].cpu().numpy(),
                target_future=output["predicted_target_latent"][0, index].cpu().numpy(),
                target_organ=output["organ_latents"][0, module_index].cpu().numpy(),
                base_forecast_normalized=float(normalized_by_horizon[horizon][index]),
                measurement_pure_state=current_tensor[0].cpu().numpy(),
                target_current_normalized=float(current_tensor[0, index]),
            )[None, :]
        if source == HEAD:
            if current is not None:
                candidate, head_abstained = _predict_head(
                    assets.heads["heads"][cell], feature, float(current), runtime=True
                )
        elif source == MULTI and current is not None:
            candidate, head_abstained = _predict_head(
                assets.multisource_heads["heads"][cell], feature, float(current), runtime=False
            )

        event_probability = None
        event_definition = None
        risk_status = None
        observed_source_value = None
        if is_event:
            observed_source_value = current
            current = _hypoxemia(current)
            event_definition = str(assets.event_heads["event_definition"])
            fresh = (
                target_state["age_hours"] is not None
                and float(target_state["age_hours"]) <= assets.max_age_hours
            )
            if cell in validated_event_heads and feature is not None and fresh:
                event_probability = _predict_event_risk(
                    assets.event_heads["heads"][cell], feature
                )
                candidate = event_probability
                risk_status = "calibrated_event_probability"
            else:
                candidate = None
                risk_status = "abstained"
        fresh = (
            target_state["age_hours"] is not None
            and float(target_state["age_hours"]) <= assets.max_age_hours
        )
        runtime_can_move = bool(
            decision["can_move"]
            and current is not None
            and candidate is not None
            and fresh
            and not head_abstained
        )
        if is_event and candidate is None:
            point = None
            runtime_source = "event_risk_abstained"
            reason = "missing_or_stale_o2sat_history"
        elif current is None:
            point = None
            runtime_source = "missing"
            reason = "no_observed_target_value"
        elif not fresh:
            point = None
            runtime_source = "stale_target_abstained"
            reason = "stale_target_anchor"
        elif head_abstained:
            point = float(current)
            runtime_source = "patient_persistence_anchor"
            reason = "continuous_head_abstained"
        elif runtime_can_move:
            point = float(candidate)
            runtime_source = "calibrated_event_risk_head" if is_event else source
            reason = None
        else:
            point = float(current)
            runtime_source = "patient_persistence_anchor"
            reason = str(contract.get("fallback_reason") or "external_validation_not_passed")
        interval_ready = bool(
            runtime_can_move
            and point is not None
            and not decision["source_changed"]
            and interval
            and interval.get("runtime_eligible")
        )
        if is_event:
            lower = None
            upper = None
            interval_status = (
                "not_applicable_calibrated_probability"
                if runtime_can_move
                else "event_risk_abstained"
            )
        elif runtime_can_move and decision["use_selector_interval"]:
            if decision["interval_method"] == "normalized_current_predicted_delta":
                interval_features = np.asarray([[float(current), abs(float(point) - float(current))]])
                radius = float(decision["normalized_radius"]) * _predict_interval_scale(decision["scale_model"], interval_features)
            else:
                radius = float(decision["radius"])
            lower, upper = float(point - radius), float(point + radius)
            interval_status = "unified_source_disjoint_calibration_90"
        elif interval_ready:
            if interval.get("runtime_interval_method") == "normalized_conformal":
                if feature is None:
                    raise ValueError(f"{cell} normalized interval requires features")
                scale = _predict_interval_scale(
                    interval["residual_scale_model"], feature
                )
                radius = float(interval["normalized_radius"]) * scale
            else:
                radius = float(interval["radius"])
            lower = float(point - radius)
            upper = float(point + radius)
            minimum, maximum = VARIABLE_BOUNDS.get(target, (None, None))
            if minimum is not None:
                lower = max(float(minimum), lower)
            if maximum is not None:
                upper = min(float(maximum), upper)
            interval_status = "externally_audited_mimic_split_conformal_90"
        elif point is None:
            lower = None
            upper = None
            interval_status = "missing_point"
        else:
            lower = None
            upper = None
            interval_status = "interval_external_validation_not_passed"
        cells.append(
            {
                "cell": cell,
                "target": target,
                "unit": "probability" if is_event else contract.get("target_unit") or VARIABLE_UNITS.get(model_target),
                "horizon_hours": horizon,
                "observed_value": current,
                "current_estimate": completion.get("value"),
                "current_estimate_source": completion.get("source"),
                "current_lower": completion.get("lower"),
                "current_upper": completion.get("upper"),
                "current_interval_status": completion.get("interval_status"),
                "current_related_inputs_observed": completion.get("inputs_observed"),
                "observed_source_value": observed_source_value,
                "measurement_age_hours": target_state["age_hours"],
                "observations": target_state["count"],
                "forecast_point": point,
                "event_risk_probability": event_probability,
                "event_definition": event_definition,
                "risk_status": risk_status,
                "abstained": bool(
                    not runtime_can_move
                    and (is_event or head_abstained or (current is not None and not fresh))
                ),
                "source_model": runtime_source,
                "source_changed": decision["source_changed"],
                "unified_source_audit_pass": decision["validation"]["promotion_allowed"],
                "source_validation_scope": (
                    "mimic_development_test_and_nwicu_transfer"
                    if decision["validation"]["promotion_allowed"] else "retained_original_release_evidence"
                ),
                "can_move": runtime_can_move,
                "fallback_reason": reason,
                "interval_status": interval_status,
                "lower": lower,
                "upper": upper,
                "causal_claim_allowed": False,
            }
        )
    return {
        "schema": "whole_body_jepa.patient_forecast.v1",
        "release_date": "2026-09-02",
        "patient": {
            "calibration_id": digest,
            "filename": Path(filename).name,
            "rows": int(len(frame)),
            "timeline_hours": float(frame["hours_since_onset"].iloc[-1] - frame["hours_since_onset"].iloc[0]),
            "observed_variables": int(sum(item["observed"] for item in state.values())),
            "total_variables": len(variables),
        },
        "cells": cells,
        "current_state": completed_state,
        "summary": {
            "catalog_cells": len(cells),
            "runtime_move_cells": int(sum(cell["can_move"] for cell in cells)),
            "fallback_cells": int(sum(not cell["can_move"] and cell["forecast_point"] is not None for cell in cells)),
            "missing_cells": int(sum(cell["forecast_point"] is None for cell in cells)),
            "nowcasted_variables": int(sum(
                item["source"] == "validated_cross_target_nowcast"
                for item in completed_state.values()
            )),
            "derived_variables": int(sum(
                item["source"] == "derived_formula"
                for item in completed_state.values()
            )),
        },
        "causal_claim_allowed": False,
        "clinical_decision_allowed": False,
    }
