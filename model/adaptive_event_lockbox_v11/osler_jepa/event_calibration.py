"""Calibration of event probabilities, its gate and its diagnostics.

Logistic recalibration (Cox 1958; Van Calster et al. 2016, 2019): the slope is
the coefficient of logit(p) in a logistic model of the outcome; calibration-in-
the-large (CITL) is the intercept with logit(p) as an offset.

Gate (fixed 2026-10-28 before any online-calibration result was seen, after the
static dev check showed that a CI-contains-0 rule is unreachable at 15,000
rows): point estimates with CITL in [-0.2, 0.2] and slope in [0.8, 1.25].
These are research tolerances, not a test of statistical equivalence.  A
stream whose calibration is not identifiable (empty, one outcome class, invalid
probabilities, a slope fit that does not converge) fails; it never raises.

Diagnostics (not gated): patient-cluster bootstrap intervals and per-group
estimates with support counts; a group is estimated only with >= 200 rows and
>= 20 events and non-events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import warnings

CITL_BOUNDS = (-0.2, 0.2)
SLOPE_BOUNDS = (0.8, 1.25)
PROBABILITY_FLOOR = 1e-6
BOOTSTRAP_SAMPLES = 200
GROUP_MIN_ROWS = 200
GROUP_MIN_CLASS = 20


def valid_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    return np.isfinite(probability) & (probability >= 0.0) & (probability <= 1.0)


def calibration(probability: np.ndarray, outcome: np.ndarray) -> dict:
    """Slope, CITL and observed/expected; ``identifiable`` False (estimates None) instead of an exception."""

    probability, outcome = np.asarray(probability, dtype=np.float64), np.asarray(outcome, dtype=np.float64)
    events = float(np.sum(outcome)) if len(outcome) else 0.0
    empty = {"identifiable": False, "slope": None, "citl": None, "observed_expected": None}
    if (len(outcome) == 0 or not np.all(valid_probability(probability)) or not np.all(np.isin(outcome, (0.0, 1.0)))
            or events == 0 or events == len(outcome)):
        return empty
    predictor = logit(np.clip(probability, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            slope = float(LogisticRegression(penalty=None, max_iter=1000).fit(predictor[:, None], outcome).coef_[0, 0])
        except ConvergenceWarning:
            return empty
    rate = events / len(outcome)
    bound = float(np.max(np.abs(predictor)) + abs(logit(rate)) + 1.0)    # the CITL root lies inside +-bound
    citl = brentq(lambda a: float(np.sum(outcome - expit(a + predictor))), -bound, bound)
    return {"identifiable": True, "slope": slope, "citl": float(citl), "observed_expected": float(rate / probability.mean())}


def calibration_pass(estimate: dict) -> bool:
    return bool(estimate["identifiable"] and CITL_BOUNDS[0] <= estimate["citl"] <= CITL_BOUNDS[1]
                and SLOPE_BOUNDS[0] <= estimate["slope"] <= SLOPE_BOUNDS[1])


def cluster_bootstrap(predictions: pd.DataFrame, column: str, *, seed: int) -> dict:
    """Percentile 95% intervals of slope and CITL, resampling whole clusters of ``column``."""

    groups = [group.index.to_numpy() for _, group in predictions.groupby(column, sort=True)]
    if len(groups) < 2:
        return {"clusters": len(groups), "slope": None, "citl": None}
    rng = np.random.default_rng(seed)
    probability, outcome = predictions["probability"].to_numpy(), predictions["outcome"].to_numpy()
    draws = [calibration(probability[rows], outcome[rows]) for rows in
             (np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(BOOTSTRAP_SAMPLES))]
    draws = [draw for draw in draws if draw["identifiable"]]
    return {"clusters": len(groups), "identifiable_draws": len(draws),
            **{key: [float(v) for v in np.quantile([d[key] for d in draws], [0.025, 0.975])] if draws else None
               for key in ("slope", "citl")}}


def group_calibration(predictions: pd.DataFrame, column: str) -> list[dict]:
    """Support counts of every group of ``column`` and its calibration where the support suffices."""

    output = []
    for key, group in predictions.groupby(column, sort=True):
        events = int(group["outcome"].sum())
        supported = len(group) >= GROUP_MIN_ROWS and min(events, len(group) - events) >= GROUP_MIN_CLASS
        estimate = calibration(group["probability"].to_numpy(), group["outcome"].to_numpy()) if supported else None
        output.append({"group": str(key), "rows": int(len(group)), "events": events, "estimate": estimate})
    return output
