"""Row-wise residual scale for normalized conformal intervals.

Normalized nonconformity scores (Papadopoulos et al. 2008; Lei et al. 2018):
score = |residual| / scale(x), with scale estimated on the calibration rows.
Schemes, all estimated from calibration residuals and inputs only:

- ``quintile``: median residual per bin of the current value, bins at the
  calibration quintiles (the v2 interval);
- ``tail8``: the same with tail-refined bins at the 2/10/25/50/75/90/98th
  percentiles, so extreme current values get their own scale;
- ``tail8_change``: ``tail8`` times a separable factor for recent volatility,
  |current - value at the previous anchor of the stay|, in four groups (no
  previous anchor, <= median, median-90th percentile, > 90th percentile).

Bins or groups with fewer than ``MIN_PATIENTS`` calibration patients use the
overall scale (factor 1 for the volatility groups).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCHEMES = ("quintile", "tail8", "tail8_change")
EDGE_QUANTILES = {
    "quintile": (0.2, 0.4, 0.6, 0.8),
    "tail8": (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
    "tail8_change": (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98),
}
MIN_PATIENTS = 30


def _group_medians(groups: np.ndarray, values: np.ndarray, subjects: np.ndarray, fallback: float) -> dict[int, float]:
    medians = {}
    for group in np.unique(groups):
        mask = groups == group
        enough = len(np.unique(subjects[mask])) >= MIN_PATIENTS
        medians[int(group)] = max(float(np.median(values[mask])), 1e-9) if enough else fallback
    return medians


@dataclass(frozen=True)
class IntervalScale:
    edges: np.ndarray
    value_scales: dict[int, float]
    overall: float
    change_edges: np.ndarray | None
    change_factors: dict[int, float]

    def change_groups(self, change: np.ndarray) -> np.ndarray:
        change = np.abs(np.asarray(change, dtype=np.float64))
        groups = np.zeros(len(change), dtype=int)
        finite = np.isfinite(change)
        groups[finite] = 1 + np.digitize(change[finite], self.change_edges)
        return groups

    def __call__(self, current: np.ndarray, change: np.ndarray) -> np.ndarray:
        bins = np.digitize(np.asarray(current, dtype=np.float64), self.edges)
        scale = np.array([self.value_scales.get(int(b), self.overall) for b in bins])
        if self.change_edges is not None:
            scale = scale * np.array([self.change_factors.get(int(g), 1.0) for g in self.change_groups(change)])
        return scale


def fit_interval_scale(scheme: str, current: np.ndarray, change: np.ndarray, residual: np.ndarray,
                       subjects: np.ndarray) -> IntervalScale:
    current = np.asarray(current, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    subjects = np.asarray(subjects).astype(str)
    overall = max(float(np.median(residual)), 1e-9)
    finite = current[np.isfinite(current)]
    edges = np.unique(np.quantile(finite, EDGE_QUANTILES[scheme]))
    value_scales = _group_medians(np.digitize(current, edges), residual, subjects, overall)
    scale = IntervalScale(edges, value_scales, overall, None, {})
    if scheme != "tail8_change":
        return scale
    change = np.asarray(change, dtype=np.float64)
    observed = np.abs(change[np.isfinite(change)])
    change_edges = np.quantile(observed, (0.5, 0.9)) if len(observed) else np.array([np.inf, np.inf])
    with_edges = IntervalScale(edges, value_scales, overall, change_edges, {})
    ratio = residual / scale(current, change)
    reference = max(float(np.median(ratio)), 1e-9)
    factors = {g: m / reference for g, m in _group_medians(with_edges.change_groups(change), ratio, subjects, reference).items()}
    return IntervalScale(edges, value_scales, overall, change_edges, factors)
