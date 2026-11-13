"""Live, per-hospital state of the online marginal M, one forecast at a time.

The frozen forms are batch replays over a whole stream
(``osler_jepa.site_adaptive_point.online_point_correction`` and
``osler_jepa.predictive_distribution.mondrian_multiqt_stream``, combined by
``jepa_deployment_form_dev.deploy_distribution``).  This module keeps the same
state between requests and applies the same updates in the same order:

- a forecast is issued from the current state and kept as pending;
- an outcome posted for it is not applied at once.  It is applied when the cell
  issues its next forecast (point correction) or its next forecast in the same
  value category (MultiQT), in issue order, exactly where the batch replay
  resolves an outcome whose availability time has passed.

So a hospital that posts each outcome once it is measured reproduces the frozen
replay (``test_osler_api_online`` checks this against the batch functions).
States are plain dictionaries of numbers so they can be stored as JSON.
Research runtime only.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import isotonic_regression

from osler_jepa.predictive_distribution import MultiQTConfig, quantile_score
from osler_jepa.site_adaptive_point import PointCorrectionConfig


def new_cell_state() -> dict:
    return {"offset": 0.0, "gain": 0.0, "categories": {}, "pending": []}


def _played(base: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    return np.array([isotonic_regression(base + offset, increasing=True) for offset in hidden])


def _category(state: dict, group: int, base: np.ndarray, config: MultiQTConfig) -> dict:
    key = str(group)
    if key not in state["categories"]:
        experts, levels = len(config.expert_etas), len(config.levels)
        state["categories"][key] = {"hidden": np.zeros((experts, levels)).tolist(),
                                    "weights": np.full(experts, 1.0 / experts).tolist()}
    return state["categories"][key]


def _resolve_point(state: dict, entry: dict, scale: float, config: PointCorrectionConfig) -> None:
    observed, issued, delta = entry["observed"], entry["point"], entry["delta"]
    if not (np.isfinite(observed) and np.isfinite(issued)):
        return
    direction = float(np.sign(observed - issued))
    low, high = config.gain_bounds
    state["offset"] += config.eta_offset * direction
    state["gain"] = float(np.clip(state["gain"] + config.eta_gain * direction * delta / scale, low, high))


def _resolve_quantiles(category: dict, entry: dict, base: np.ndarray, config: MultiQTConfig) -> None:
    observed, prediction = entry["observed"], entry["point"]
    if not (np.isfinite(observed) and np.isfinite(prediction)):
        return
    levels, etas = np.asarray(config.levels, dtype=np.float64), np.asarray(config.expert_etas, dtype=np.float64)
    hidden, weights = np.asarray(category["hidden"]), np.asarray(category["weights"])
    current = _played(base, hidden)
    z = (observed - prediction) / max(entry["scale"], 1e-9)
    covered = (z <= current).astype(np.float64)
    hidden = hidden - etas[:, None] * (covered - levels[None, :])
    log_weights = np.log(weights) - config.meta_rate * quantile_score(z, current, levels)
    updated = np.exp(log_weights - log_weights.max())
    weights = (1.0 - config.fixed_share) * updated / updated.sum() + config.fixed_share / len(etas)
    category["hidden"], category["weights"] = hidden.tolist(), weights.tolist()


def issue(state: dict, *, key: str, current: float, prediction: float, group: int, scale: float, base: np.ndarray,
          point_scale: float, point_config: PointCorrectionConfig, multiqt: MultiQTConfig) -> tuple[float, np.ndarray]:
    """Apply the outcomes posted since the cell's last forecast, then issue the corrected point and the normalized
    quantiles of one row; the row is kept as pending under ``key``.  ``scale``: the row's distribution scale;
    ``base``: its category's day-one normalized quantiles; ``point_scale``: the cell's point-correction scale."""

    if point_config.level_etas:
        raise ValueError("per-level point offsets are not part of the frozen M form")
    point_scale = max(float(point_scale), 1e-9)
    for entry in state["pending"]:
        if entry["observed"] is not None and not entry["point_applied"]:
            _resolve_point(state, entry, point_scale, point_config)
            entry["point_applied"] = True
    category = _category(state, group, base, multiqt)
    for entry in state["pending"]:
        if entry["observed"] is not None and not entry["quantiles_applied"] and entry["group"] == group:
            _resolve_quantiles(category, entry, base, multiqt)
            entry["quantiles_applied"] = True
    state["pending"] = [e for e in state["pending"] if not (e["point_applied"] and e["quantiles_applied"])]
    delta = prediction - current
    if np.isfinite(current) and np.isfinite(delta):
        point = current + point_scale * state["offset"] + (1.0 + state["gain"]) * delta
    else:
        point = prediction
    normalized = np.asarray(category["weights"]) @ _played(base, np.asarray(category["hidden"]))
    state["pending"].append({"key": key, "group": int(group), "point": float(point), "delta": float(delta),
                             "scale": float(scale), "observed": None, "point_applied": False, "quantiles_applied": False})
    return float(point), normalized


def post_outcome(state: dict, key: str, observed: float) -> bool:
    """Record the outcome of the pending forecast ``key``; it is applied at the cell's next forecast.  Returns False
    when no pending forecast has that key or it already has an outcome."""

    for entry in state["pending"]:
        if entry["key"] == key and entry["observed"] is None:
            entry["observed"] = float(observed)
            return True
    return False


def expire(state: dict, keys: set[str]) -> None:
    """Drop pending forecasts that never received an outcome (they would never update the state)."""

    state["pending"] = [e for e in state["pending"] if not (e["key"] in keys and e["observed"] is None)]
