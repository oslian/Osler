"""Online Platt scaling of a frozen event probability at a new site.

Gupta & Ramdas (2023), "Online Platt Scaling with Calibeating", Algorithm 1:
the recalibrated probability is ``sigmoid(a * logit(p) + b)``; ``(a, b)``
starts at ``(1, 0)`` (the frozen model unchanged) and is learned by the Online
Newton Step (Hazan 2016, Algorithm 12) on log loss with the paper's settings
gamma = 0.1, A0 = rho * I (rho = 100) and the domain ||(a, b)|| <= 100.

Causality matches ``site_adaptive_point``: an outcome updates ``(a, b)`` only
once the shared deployment clock passes its resolve time.  A row whose input
probability is not finite or lies outside [0, 1] gets NaN and never updates
``(a, b)``; neither does a row with a missing outcome.  Inputs are clipped to
[1e-6, 1 - 1e-6] before the logit.  The frozen model's parameters are never
changed.  Factual research-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from osler_jepa.event_calibration import PROBABILITY_FLOOR, valid_probability


@dataclass(frozen=True)
class OnlinePlattConfig:
    gamma: float = 0.1
    rho: float = 100.0
    radius: float = 100.0


def project(theta: np.ndarray, matrix: np.ndarray, radius: float) -> np.ndarray:
    """argmin over ||x|| <= radius of (theta - x)' A (theta - x)."""

    if np.linalg.norm(theta) <= radius:
        return theta
    solve = lambda lam: np.linalg.solve(matrix + lam * np.eye(2), matrix @ theta)
    lam = brentq(lambda lam: np.linalg.norm(solve(lam)) - radius, 0.0, 1e12)
    return solve(lam)


def online_platt_scaling(*, anchor_hours: np.ndarray, resolve_hours: np.ndarray, probability: np.ndarray,
                         outcome: np.ndarray, config: OnlinePlattConfig) -> np.ndarray:
    """Causally recalibrated probabilities for one site's stream in clock order."""

    probability = np.asarray(probability, dtype=np.float64)
    valid = valid_probability(probability)
    score = logit(np.clip(np.where(valid, probability, 0.5), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR))
    theta = np.array([1.0, 0.0])
    matrix = config.rho * np.eye(2)
    calibrated = np.empty(len(score))
    pending: list[int] = []
    for row in range(len(score)):
        still_pending = []
        for earlier in pending:
            if resolve_hours[earlier] <= anchor_hours[row] and valid[earlier] and np.isfinite(outcome[earlier]):
                features = np.array([score[earlier], 1.0])
                gradient = (expit(theta @ features) - outcome[earlier]) * features
                matrix += np.outer(gradient, gradient)
                theta = project(theta - np.linalg.solve(matrix, gradient) / config.gamma, matrix, config.radius)
            elif resolve_hours[earlier] > anchor_hours[row]:
                still_pending.append(earlier)
        pending = still_pending
        calibrated[row] = expit(theta[0] * score[row] + theta[1]) if valid[row] else np.nan
        pending.append(row)
    return calibrated
