"""The derived threshold-event deployment form: clinical thresholds read from the Mondrian predictive distribution.

Shared by the MIMIC development (``predictive_distribution_all_dev.py``) and the lockbox scorer
(``site_adaptive_threshold_audit.py``), so both compute exactly the same form:

- the point forecast is the cell's frozen recipe with the frozen online point correction;
- day one: signed residuals of held-out MIMIC conformal patients, normalized by the v2 quintile scale fitted on the
  selection patients, weighted for covariate shift against the deployment's warm-up inputs;
- online: ``osler_jepa.predictive_distribution.mondrian_multiqt_stream`` with categories = ``tail8`` bins of the
  current value (edges from held-out MIMIC patients);
- a threshold probability is read from that distribution with a continuity rule: with recording resolution r (the
  largest of 1, 0.1, 0.01 of which >= 99% of the deployment's current values are multiples; label-free; else 0),
  P(Y <= c) = F(c + r/2), P(Y < c) = F(c - r/2), upper-tail events are the complements, and a rise of at least d
  from the current value x is 1 - F(x + d - r/2).

Clinical thresholds were approved by the user on 2026-10-29.  Factual research-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from target_specific_domain_conformal_audit import _safe_float
from osler_jepa.interval_scale import fit_interval_scale
from osler_jepa.patient_split import selection_patients
from osler_jepa.predictive_distribution import MondrianStream, MultiQTConfig, mondrian_multiqt_stream, value_bins
from osler_jepa.site_adaptive_conformal import covariate_shift_weights
from osler_jepa.site_adaptive_point import PointCorrectionConfig, online_point_correction
from site_adaptive_interval_dev import Replay, covariate_matrix

SCALE_SCHEME = "quintile"
GLUCOSE = (("gt", 180.0), ("lt", 70.0))
THRESHOLDS = {
    "potassium": (("ge", 6.0), ("lt", 3.5)),
    "sodium": (("lt", 130.0), ("gt", 150.0)),
    "map": (("lt", 65.0),),
    "heart_rate": (("gt", 120.0), ("lt", 50.0)),
    "respiratory_rate": (("gt", 30.0),),
    "glucose": GLUCOSE,
    "glucose_poc": GLUCOSE,
    "hemoglobin": (("lt", 7.0),),
    "platelets": (("lt", 50.0),),
    "creatinine": (("rise", 0.3),),
    "bicarbonate": (("lt", 18.0), ("lt", 15.0)),
    "o2sat": (("le", 92.0),),
}
MIN_CLASS = 20
INTEGER_SHARE = 0.99


def label(op: str, value: float) -> str:
    return f"{op}_{value:g}"


def resolution(current: np.ndarray) -> float:
    values = current[np.isfinite(current)]
    for step in (1.0, 0.1, 0.01):
        if np.mean(np.isclose(values / step, np.round(values / step), atol=1e-6)) >= INTEGER_SHARE:
            return step
    return 0.0


def event_of(op: str, value: float, future: np.ndarray, current: np.ndarray) -> np.ndarray:
    eps = 1e-9
    return {"le": future <= value + eps, "lt": future < value - eps, "ge": future >= value - eps,
            "gt": future > value + eps, "rise": future - current >= value - eps}[op].astype(float)


def persistence_of(op: str, value: float, current: np.ndarray) -> np.ndarray:
    """The event evaluated on the current value; a rise from the current value is never present now."""

    return np.zeros(len(current)) if op == "rise" else event_of(op, value, current, current)


def event_probability(distribution: MondrianStream, op: str, value: float, current: np.ndarray, step: float) -> np.ndarray:
    half = step / 2.0
    if op == "le":
        return distribution.probability(value + half)
    if op == "lt":
        return distribution.probability(value - half)
    if op == "ge":
        return 1.0 - distribution.probability(value - half)
    if op == "gt":
        return 1.0 - distribution.probability(value + half)
    return 1.0 - distribution.probability(current + value - half)


def calibration_state(frame: pd.DataFrame, fitted: dict, target: str) -> dict:
    """Day-one ingredients from held-out MIMIC patients: scale, signed normalized scores and Mondrian edges."""

    future, current = _safe_float(frame, f"future_{target}"), _safe_float(frame, f"{target}_t")
    change = np.abs(current - _safe_float(frame, f"lag1__{target}_t"))
    held_out = fitted["calibration_rows"]
    residual = future[held_out] - fitted["calibration_prediction"]
    choosing = selection_patients(frame.iloc[held_out]["subject_id"].to_numpy())
    selected, conformal = held_out[choosing], held_out[~choosing]
    scale = fit_interval_scale(SCALE_SCHEME, current[selected], change[selected], np.abs(residual[choosing]),
                               frame.iloc[selected]["subject_id"].to_numpy())
    return {"scale": scale, "conformal": conformal, "signed": residual[~choosing] / scale(current[conformal], change[conformal]),
            "point_scale": float(np.median(np.abs(residual[choosing]))), "edges": value_bins(current[held_out])}


def deploy(frame: pd.DataFrame, deployment: Replay, state: dict, point: pd.Series, target: str, *,
           point_config: PointCorrectionConfig, multiqt: MultiQTConfig) -> tuple[np.ndarray, MondrianStream]:
    """Corrected point forecast and Mondrian distribution of one deployment's rows (in clock order)."""

    rows, conformal = deployment.rows, state["conformal"]
    future, current = _safe_float(frame, f"future_{target}"), _safe_float(frame, f"{target}_t")
    change = np.abs(current - _safe_float(frame, f"lag1__{target}_t"))
    weights = covariate_shift_weights(covariate_matrix(frame, conformal, target, frame.iloc[conformal]),
                                      covariate_matrix(frame, deployment.day_one, target, frame.iloc[conformal]))
    common = deployment.common(frame, {"care_unit": "careunit"})
    scales = state["scale"](current[rows], change[rows])
    corrected = online_point_correction(current=current[rows], prediction=point[rows].to_numpy(), observed=future[rows],
                                        scale=state["point_scale"], config=point_config, **common)
    edges = state["edges"]
    distribution = mondrian_multiqt_stream(
        calibration_scores=state["signed"], calibration_weights=weights, calibration_groups=np.digitize(current[conformal], edges),
        calibration_subjects=frame.iloc[conformal]["subject_id"].to_numpy(), groups=np.digitize(current[rows], edges),
        predictions=corrected, scales=scales, observed=future[rows], anchor_hours=deployment.clock,
        resolve_hours=deployment.resolve, config=multiqt)
    return corrected, distribution
