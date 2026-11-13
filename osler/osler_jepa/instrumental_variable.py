"""Instrumental-variable estimation and readiness checks for preference-based designs.

Two-stage least squares (2SLS) with exogenous covariates, cluster bootstrap
confidence intervals (resampling whole clusters, e.g. hospitals, because a
preference instrument is defined at that level), and the diagnostics a
preference-based instrument needs (Brookhart & Schneeweiss 2007; Baiocchi,
Cheng & Small 2014):

- instrument strength: first-stage F >= 10 (Stock & Yogo 2005);
- covariate balance across instrument levels;
- falsification: an outcome measured before treatment (placebo) and a negative
  control outcome must show no effect;
- benchmark: an effect known from randomized trials must be reproduced.

No IV library is installed in this environment (PEP 668 system Python), so
2SLS uses its closed form with numpy; ``test_instrumental_variable.py``
checks it on data with a known effect.  A passing readiness result permits a
research causal-effect estimate only after a documented external review; it
never grants clinical or runtime authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _design(covariates: np.ndarray | None, n: int) -> np.ndarray:
    constant = np.ones((n, 1))
    if covariates is None or covariates.size == 0:
        return constant
    return np.column_stack([constant, np.asarray(covariates, dtype=np.float64)])


def two_stage_least_squares(outcome, treatment, instrument, covariates=None) -> float:
    """Coefficient on ``treatment`` from 2SLS with one instrument and exogenous covariates."""

    y = np.asarray(outcome, dtype=np.float64)
    d = np.asarray(treatment, dtype=np.float64)
    z = np.asarray(instrument, dtype=np.float64)
    w = _design(covariates, len(y))
    instruments = np.column_stack([z, w])
    regressors = np.column_stack([d, w])
    first, *_ = np.linalg.lstsq(instruments, regressors, rcond=None)
    fitted = instruments @ first
    beta, *_ = np.linalg.lstsq(fitted, y, rcond=None)
    return float(beta[0])


def first_stage_f(treatment, instrument, covariates=None) -> float:
    """F statistic for the excluded instrument in the first stage (one instrument: t squared)."""

    d = np.asarray(treatment, dtype=np.float64)
    z = np.asarray(instrument, dtype=np.float64)
    w = _design(covariates, len(d))
    full = np.column_stack([z, w])
    coef, *_ = np.linalg.lstsq(full, d, rcond=None)
    residual = d - full @ coef
    dof = len(d) - full.shape[1]
    sigma2 = float(residual @ residual) / dof
    inverse = np.linalg.pinv(full.T @ full)
    se = np.sqrt(sigma2 * inverse[0, 0])
    return float((coef[0] / se) ** 2)


def standardized_mean_difference(values, group) -> float:
    values = np.asarray(values, dtype=np.float64)
    group = np.asarray(group, dtype=bool)
    left, right = values[~group], values[group]
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2.0)
    return float((right.mean() - left.mean()) / pooled) if pooled > 0 else 0.0


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    low: float
    high: float
    replicates: int


def cluster_bootstrap(statistic, clusters, *, replicates: int = 500, seed: int = 0) -> BootstrapEstimate:
    """Percentile 95% CI of ``statistic(row_indices)`` resampling whole clusters with replacement."""

    clusters = np.asarray(clusters)
    labels, codes = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(codes == k) for k in range(len(labels))]
    rng = np.random.default_rng(seed)
    estimate = float(statistic(np.arange(len(clusters))))
    draws = []
    for _ in range(replicates):
        picked = rng.integers(0, len(labels), size=len(labels))
        rows = np.concatenate([members[k] for k in picked])
        value = statistic(rows)
        if np.isfinite(value):
            draws.append(value)
    low, high = np.quantile(draws, [0.025, 0.975])
    return BootstrapEstimate(estimate, float(low), float(high), len(draws))
