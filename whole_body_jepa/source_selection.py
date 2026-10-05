"""Frozen JEPA readouts and patient-disjoint, development-only source selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .multisource_intervals import (
    MIN_PATIENTS, SEEDS, _bootstrap, coverage, domain_coverage,
    fit_radius, patient_split, point_gates, sha256, stable_hash,
)

BASE = "frozen_whole_body_jepa"
HEAD = "unified_shared_jepa_target_head"
MULTI = "multisource_shared_jepa_target_head"
PERSISTENCE = "patient_persistence_anchor"
SOURCES = (BASE, HEAD, MULTI, PERSISTENCE)
BOUND_FILES = (
    "whole_body_joint_jepa.pt", "FEATURE_CONTRACT.json",
    "unified_shared_jepa_head_bundle.joblib", "multisource_shared_jepa_head_bundle.joblib",
    "runtime_registry.json", "interval_calibrator.json",
)


def predict_head_result(
    head: dict,
    features: np.ndarray,
    current: np.ndarray,
    *,
    runtime: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical predictions and the head's explicit abstention mask."""
    from osler_jepa.shared_release_features import FEATURE_SLICES, FEATURE_WIDTH

    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
        raise ValueError("source features must follow the 311-column contract")
    view = head.get("feature_view", "all")
    if view == "shared":
        features = features[:, :FEATURE_SLICES["base_point"].stop]
    elif view == "measurement":
        features = features[:, FEATURE_SLICES["base_point"].start:]
    elif view != "all":
        raise ValueError(f"unsupported feature view: {view}")
    if features.shape[1] != head["feature_width"]:
        raise ValueError("source head feature width mismatch")
    prefix = "runtime_feature_" if runtime else "feature_"
    strategy = head.get("feature_normalization", "robust")
    if strategy == "robust":
        dtype = np.float64 if runtime else None
        transformed = (features - np.asarray(head[prefix + "median"], dtype=dtype)) / np.asarray(head[prefix + "scale"], dtype=dtype)
    elif strategy == "quantile":
        transformed = head[prefix + "quantile_transformer"].transform(features)
    else:
        raise ValueError(f"unsupported normalization: {strategy}")
    if head["family"] == "ridge":
        delta = transformed @ np.asarray(head["coefficient"]) + float(head["intercept"])
    elif head["family"] == "hurdle":
        delta = head["classifier"].predict_proba(transformed)[:, 1] * head["regressor"].predict(transformed)
    elif head["family"] == "robust_tree":
        delta = head["regressor"].predict(transformed)
    else:
        raise ValueError("unsupported head family")
    threshold = float(head.get("normalized_abstention_threshold", 0))
    abstained = np.abs(delta) < threshold
    delta = np.where(abstained, 0, delta)
    limit = float(head["normalized_delta_limit"])
    result = current + float(head["shrinkage"]) * np.clip(delta, -limit, limit) * float(head["target_scale"])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite source prediction")
    return result, abstained


def predict_head_values(head: dict, features: np.ndarray, current: np.ndarray, *, runtime: bool) -> np.ndarray:
    values, _ = predict_head_result(head, features, current, runtime=runtime)
    return values


def patient_roles(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    codes = np.asarray([stable_hash("source_roles_v1", s) % 10 for s in frame["subject_id"].astype(str)])
    return {"selection": codes < 4, "calibration": (codes >= 4) & (codes < 7), "test": codes >= 7}


def residual_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    output = frame[["subject_id", "stay_id", "t", "careunit", "current", "truth"]].copy()
    output["residual"] = np.abs(frame["truth"].to_numpy() - frame[source].to_numpy())
    return output


def paired(frame: pd.DataFrame, source: str, baseline: str, salt: str) -> dict:
    delta = np.abs(frame["truth"] - frame[source]) - np.abs(frame["truth"] - frame[baseline])
    values = delta.groupby(frame["subject_id"]).mean().to_numpy()
    if len(values) < MIN_PATIENTS:
        return {"pass": False, "available": False, "reason": "insufficient_patients"}
    return _bootstrap(values, stable_hash(salt))


def select_source(selection: pd.DataFrame, incumbent: str, cell: str) -> dict:
    """Never receives calibration, final test or external labels."""
    scores = {}
    for source in SOURCES:
        if source not in selection:
            continue
        error = (selection["truth"] - selection[source]).abs()
        mae = float(error.groupby(selection["subject_id"]).mean().mean()) if len(selection) else None
        folds = []
        if source != incumbent:
            for seed in SEEDS:
                _, heldout = patient_split(selection["subject_id"], seed)
                fold = selection.loc[heldout]
                folds.append({"seed": seed,
                              "vs_incumbent": paired(fold, source, incumbent, f"{cell}:{source}:{seed}:incumbent"),
                              "vs_persistence": paired(fold, source, PERSISTENCE, f"{cell}:{source}:{seed}:persistence")})
        eligible = bool(source != incumbent and folds and all(
            f["vs_incumbent"]["pass"] and f["vs_persistence"]["pass"] for f in folds))
        scores[source] = {"patient_equal_mae": mae, "seven_split_comparisons": folds, "eligible_challenger": eligible}
    eligible = [s for s in scores if scores[s]["eligible_challenger"]]
    winner = min(eligible, key=lambda s: (scores[s]["patient_equal_mae"], SOURCES.index(s))) if eligible else incumbent
    return {"selected_source": winner, "incumbent_source": incumbent, "scores": scores,
            "selection_rule": "lowest_patient_equal_mae_among_7_split_significant_challengers_else_incumbent"}


def compare_on_holdout(frame: pd.DataFrame, source: str, incumbent: str, cell: str) -> dict:
    residuals = residual_frame(frame, source)
    persistence = point_gates(residuals, cell)
    if source == incumbent:
        against_incumbent = {"pass": True, "reason": "same_frozen_source"}
    else:
        competitor = residuals.copy()
        competitor["current"] = frame[incumbent].to_numpy()
        against_incumbent = point_gates(competitor, cell + ":vs_incumbent")
    coverage_input = residuals
    truth_delta = frame["truth"].to_numpy() - frame["current"].to_numpy()
    pred_delta = frame[source].to_numpy() - frame["current"].to_numpy()
    correlation = float(np.corrcoef(truth_delta, pred_delta)[0, 1]) if (
        len(frame) > 1 and np.std(truth_delta) > 0 and np.std(pred_delta) > 0) else None
    return {"vs_persistence": persistence, "vs_incumbent": against_incumbent,
            "pass": persistence["pass"] and against_incumbent["pass"],
            "direction_accuracy": float(np.mean(np.sign(truth_delta) == np.sign(pred_delta))),
            "delta_correlation": correlation,
            "patient_equal_mae": float(coverage_input.groupby("subject_id")["residual"].mean().mean())}


def validate_selected(development: pd.DataFrame, external: pd.DataFrame, selection: dict, cell: str) -> dict:
    roles = patient_roles(development)
    source, incumbent = selection["selected_source"], selection["incumbent_source"]
    support = {role: int(development.loc[mask, "subject_id"].nunique()) for role, mask in roles.items()}
    if set(development["subject_id"]) & set(external["subject_id"]):
        raise ValueError("development and external patients overlap")
    if min(support.values()) < MIN_PATIENTS:
        return {"promotion_allowed": False, "radius": None, "support": support,
                "failure_reasons": ["insufficient_unused_development_patients"]}
    cal = residual_frame(development.loc[roles["calibration"]], source)
    test = development.loc[roles["test"]]
    radius = fit_radius(cal, SEEDS[0])
    results = {}
    for name, frame in (("mimic_test", test), ("nwicu", external)):
        residuals = residual_frame(frame, source)
        times = pd.to_datetime(frame["t"])
        results[name] = {"point": compare_on_holdout(frame, source, incumbent, cell),
                         "coverage": coverage(residuals, radius),
                         "careunit_coverage": domain_coverage(residuals, radius),
                         "late_coverage": coverage(residuals.loc[times >= times.quantile(.8)], radius)}
    reasons = [f"{name}:{gate}" for name, gates in results.items() for gate, result in gates.items() if not result["pass"]]
    return {"promotion_allowed": not reasons, "radius": radius, "support": support,
            "failure_reasons": reasons, **results}


def load_selector(path: Path, model_dir: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "whole_body_jepa.unified_source_selector.v1":
        raise ValueError("unsupported unified source selector")
    if set(payload.get("release_binding", {})) != set(BOUND_FILES):
        raise ValueError("incomplete source selector binding")
    for name, digest in payload["release_binding"].items():
        if sha256(model_dir / name) != digest:
            raise ValueError(f"source selector binding mismatch: {name}")
    for row in payload["cells"].values():
        if row["source_model"] not in SOURCES:
            raise ValueError("unknown selected source")
        if row["source_changed"] != (row["source_model"] != row["incumbent_source"]):
            raise ValueError("inconsistent source replacement")
        if row["source_model"] == PERSISTENCE and row["can_move"]:
            raise ValueError("persistence cannot be marked movable")
        if row["source_changed"] and not row["validation"]["promotion_allowed"]:
            raise ValueError("unvalidated source replacement")
        if row["use_selector_interval"]:
            method = row["interval_method"]
            if method == "constant":
                radius = row["radius"]
            elif method == "normalized_current_predicted_delta":
                radius = row["normalized_radius"]
                model = row["scale_model"]
                if len(model["center"]) != 2 or len(model["scale"]) != 2 or len(model["beta"]) != 3:
                    raise ValueError("conditional interval must use current and predicted delta")
                if not all(np.isfinite(model[k]).all() for k in ("center", "scale", "beta")) or min(model["scale"]) <= 0:
                    raise ValueError("invalid conditional interval scale")
                if not (np.isfinite(model["lower"]) and np.isfinite(model["upper"]) and 0 < model["lower"] <= model["upper"]):
                    raise ValueError("invalid conditional interval bounds")
            else:
                raise ValueError("unsupported selector interval method")
            if not row["validation"]["promotion_allowed"] or radius is None or not np.isfinite(radius) or radius < 0:
                raise ValueError("unvalidated selector interval")
    return payload
