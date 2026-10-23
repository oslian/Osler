"""Online recalibration of a frozen point forecast at a new site.

The frozen forecast is ``current + delta``.  At a new site it is recalibrated
online as outcomes arrive::

    corrected = current + scale * (a_site + a_level[key]) + (1 + b_site) * delta

``a`` (offset) and ``b`` (change gain) are updated by online subgradient
descent on absolute error in units of ``scale`` (online median regression).
``b`` is clipped to [-1, b_max]: -1 recovers persistence, so a site where the
learned dynamics do not transfer can fall back towards persistence by itself.
Additional offset levels (for example care unit) use site-local keys.

Causality matches ``site_adaptive_conformal``: an outcome is used only after
its resolve time has passed or the stream has moved to a later stay.
Factual research-only; the frozen model's parameters are never changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PointCorrectionConfig:
    eta_offset: float = 0.0
    eta_gain: float = 0.0
    level_etas: Mapping[str, float] = field(default_factory=dict)
    min_level_history: int = 20
    gain_bounds: tuple[float, float] = (-1.0, 0.5)


def online_point_correction(
    *,
    stays: np.ndarray,
    levels: Mapping[str, np.ndarray],
    anchor_hours: np.ndarray,
    resolve_hours: np.ndarray,
    current: np.ndarray,
    prediction: np.ndarray,
    observed: np.ndarray,
    scale: float,
    config: PointCorrectionConfig,
) -> np.ndarray:
    """Return causally corrected predictions for one site's stream in arrival order."""

    count = len(stays)
    scale = max(float(scale), 1e-9)
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    offset, gain = 0.0, 0.0
    level_offset: dict[str, dict[object, float]] = {name: {} for name in config.level_etas}
    level_seen: dict[str, dict[object, int]] = {name: {} for name in config.level_etas}
    corrected = np.full(count, np.nan)
    pending: list[tuple[float, object, int]] = []
    low, high = config.gain_bounds

    def resolve(row: int) -> None:
        nonlocal offset, gain
        if not (np.isfinite(observed[row]) and np.isfinite(corrected[row])):
            return
        direction = float(np.sign(observed[row] - corrected[row]))
        offset += config.eta_offset * direction
        gain = float(np.clip(gain + config.eta_gain * direction * delta[row] / scale, low, high))
        for name, eta in config.level_etas.items():
            key = levels[name][row]
            level_offset[name][key] = level_offset[name].get(key, 0.0) + eta * direction
            level_seen[name][key] = level_seen[name].get(key, 0) + 1

    for row in range(count):
        still_pending = []
        for resolve_time, stay, pending_row in pending:
            if resolve_time <= anchor_hours[row]:   # shared deployment clock: resolve only when available
                resolve(pending_row)
            else:
                still_pending.append((resolve_time, stay, pending_row))
        pending = still_pending
        total_offset = offset
        for name in config.level_etas:
            key = levels[name][row]
            if level_seen[name].get(key, 0) >= config.min_level_history:
                total_offset += level_offset[name][key]
        if np.isfinite(current[row]) and np.isfinite(delta[row]):
            corrected[row] = current[row] + scale * total_offset + (1.0 + gain) * delta[row]
        else:
            corrected[row] = prediction[row]
        pending.append((float(resolve_hours[row]), stays[row], row))
    return corrected
