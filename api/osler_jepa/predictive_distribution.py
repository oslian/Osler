"""Online predictive distribution around a frozen point forecast.

Day one: a split conformal predictive system (Vovk et al. 2020) on signed
normalized residuals z = (y - prediction) / scale of held-out source patients,
weighted for covariate shift (``site_adaptive_conformal.weighted_quantile``),
gives base quantiles of z on a fixed level grid.

Online: MultiQT (Ding, Gibbs and Tibshirani, arXiv 2512.23671) tracks one
hidden offset per level; the played normalized quantiles are the isotonic
projection (PAVA) of base + offset, so levels never cross.  When an outcome
resolves on the shared deployment clock, each level's hidden offset takes a
quantile-tracking step on its coverage indicator, judged against the current
played quantile (the delayed-feedback rule of ``dtaci_stream_widths``).  Several
step sizes run as experts mixed by exponential weights on the quantile score
with fixed share (as DtACI); a convex combination of non-decreasing vectors is
non-decreasing, so the issued quantiles never cross either.

Threshold probabilities come from the same distribution: linear interpolation
of the CDF between grid levels; beyond the outer levels, the day-one tail shape
(weighted empirical CDF of z) shifted by the outer level's online offset and
scaled to meet that level exactly.

Mondrian version (Vovk et al.; Boström et al. 2023): rows are split into
categories known at forecast time (bins of the current value, the ``tail8``
edges of ``interval_scale``), and each category gets its own day-one shape and
its own online state, so calibration holds within each category; a category
with fewer than ``MIN_PATIENTS`` calibration patients starts from the pooled
shape.  Factual research-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import isotonic_regression

from osler_jepa.interval_scale import EDGE_QUANTILES, MIN_PATIENTS
from osler_jepa.site_adaptive_conformal import weighted_quantile


@dataclass(frozen=True)
class MultiQTConfig:
    levels: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
    expert_etas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
    meta_rate: float = 5.0
    fixed_share: float = 0.005


@dataclass(frozen=True)
class DistributionStream:
    """Issued normalized quantiles per row (rows x levels) and what is needed to read the distribution."""

    levels: np.ndarray
    normalized: np.ndarray
    base: np.ndarray
    predictions: np.ndarray
    scales: np.ndarray
    calibration_scores: np.ndarray
    calibration_weights: np.ndarray

    def quantiles(self) -> np.ndarray:
        return self.predictions[:, None] + self.scales[:, None] * self.normalized

    def quantile(self, level: float) -> np.ndarray:
        index = int(np.flatnonzero(np.isclose(self.levels, level))[0])
        return self.predictions + self.scales * self.normalized[:, index]

    def probability(self, threshold: float | np.ndarray) -> np.ndarray:
        """P(Y <= threshold) for every row; ``threshold`` is a number or one value per row."""

        z = (threshold - self.predictions) / self.scales
        levels, u = self.levels, self.normalized
        inside = np.array([np.interp(value, row, levels) for value, row in zip(z, u)])
        lower = levels[0] * self._tail(z - (u[:, 0] - self.base[0])) / max(self._tail(self.base[:1])[0], 1e-12)
        upper_mass = 1.0 - self._tail(self.base[-1:])[0]
        upper = 1.0 - (1.0 - levels[-1]) * (1.0 - self._tail(z - (u[:, -1] - self.base[-1]))) / max(upper_mass, 1e-12)
        result = np.where(z < u[:, 0], np.minimum(lower, levels[0]), np.where(z > u[:, -1], np.maximum(upper, levels[-1]), inside))
        return np.clip(result, 0.0, 1.0)

    def _tail(self, values: np.ndarray) -> np.ndarray:
        """Weighted empirical CDF of the day-one calibration scores."""

        order = np.argsort(self.calibration_scores, kind="stable")
        scores, weights = self.calibration_scores[order], self.calibration_weights[order]
        cumulative = np.cumsum(weights) / weights.sum()
        index = np.searchsorted(scores, np.asarray(values, dtype=np.float64), side="right")
        return np.where(index > 0, cumulative[np.maximum(index - 1, 0)], 0.0)


def base_quantiles(scores: np.ndarray, weights: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.array([weighted_quantile(scores, weights, float(level)) for level in levels])


def quantile_score(z: float, quantiles: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Mean pinball loss over the levels (a discretized CRPS); rows of ``quantiles`` are experts."""

    gap = z - quantiles
    return np.mean(np.where(gap >= 0, levels * gap, (levels - 1.0) * gap), axis=-1)


def multiqt_stream(*, calibration_scores: np.ndarray, calibration_weights: np.ndarray, predictions: np.ndarray,
                   scales: np.ndarray, observed: np.ndarray, anchor_hours: np.ndarray, resolve_hours: np.ndarray,
                   config: MultiQTConfig) -> DistributionStream:
    """Issue each row's normalized quantiles; an outcome updates the state only once its resolve time has passed."""

    levels = np.asarray(config.levels, dtype=np.float64)
    etas = np.asarray(config.expert_etas, dtype=np.float64)
    scales = np.maximum(np.asarray(scales, dtype=np.float64), 1e-9)
    base = base_quantiles(calibration_scores, calibration_weights, levels)
    hidden = np.zeros((len(etas), len(levels)))
    weights = np.full(len(etas), 1.0 / len(etas))
    issued = np.full((len(predictions), len(levels)), np.nan)
    pending: list[tuple[float, int]] = []

    def played() -> np.ndarray:
        return np.array([isotonic_regression(base + offset, increasing=True) for offset in hidden])

    current = played()

    def resolve(row: int) -> None:
        nonlocal hidden, weights, current
        if not (np.isfinite(observed[row]) and np.isfinite(predictions[row])):
            return
        z = (observed[row] - predictions[row]) / scales[row]
        covered = (z <= current).astype(np.float64)
        hidden = hidden - etas[:, None] * (covered - levels[None, :])
        log_weights = np.log(weights) - config.meta_rate * quantile_score(z, current, levels)
        updated = np.exp(log_weights - log_weights.max())
        weights = (1.0 - config.fixed_share) * updated / updated.sum() + config.fixed_share / len(etas)
        current = played()

    for row in range(len(predictions)):
        still_pending = []
        for resolve_time, pending_row in pending:
            if resolve_time <= anchor_hours[row]:   # shared deployment clock: resolve only when available
                resolve(pending_row)
            else:
                still_pending.append((resolve_time, pending_row))
        pending = still_pending
        issued[row] = weights @ current
        pending.append((float(resolve_hours[row]), row))
    return DistributionStream(levels=levels, normalized=issued, base=base, predictions=np.asarray(predictions, dtype=np.float64),
                              scales=scales, calibration_scores=np.asarray(calibration_scores, dtype=np.float64),
                              calibration_weights=np.asarray(calibration_weights, dtype=np.float64))


@dataclass(frozen=True)
class MondrianStream:
    """One ``DistributionStream`` per category, read back in the original row order."""

    rows: int
    parts: tuple[tuple[np.ndarray, DistributionStream], ...]

    @property
    def levels(self) -> np.ndarray:
        return self.parts[0][1].levels

    def _assemble(self, read) -> np.ndarray:
        first = read(self.parts[0][1])
        output = np.full((self.rows, *first.shape[1:]), np.nan)
        for index, part in self.parts:
            output[index] = read(part)
        return output

    def quantiles(self) -> np.ndarray:
        return self._assemble(lambda part: part.quantiles())

    def quantile(self, level: float) -> np.ndarray:
        return self._assemble(lambda part: part.quantile(level))

    def probability(self, threshold: float | np.ndarray) -> np.ndarray:
        """P(Y <= threshold); ``threshold`` is a number or one value per row."""

        per_row = np.broadcast_to(np.asarray(threshold, dtype=np.float64), (self.rows,))
        output = np.full(self.rows, np.nan)
        for index, part in self.parts:
            output[index] = part.probability(per_row[index])
        return output


def value_bins(current: np.ndarray) -> np.ndarray:
    """Category edges: the ``tail8`` calibration quantiles of the current value."""

    finite = np.asarray(current, dtype=np.float64)
    return np.unique(np.quantile(finite[np.isfinite(finite)], EDGE_QUANTILES["tail8"]))


def mondrian_multiqt_stream(*, calibration_scores: np.ndarray, calibration_weights: np.ndarray,
                            calibration_groups: np.ndarray, calibration_subjects: np.ndarray, groups: np.ndarray,
                            predictions: np.ndarray, scales: np.ndarray, observed: np.ndarray, anchor_hours: np.ndarray,
                            resolve_hours: np.ndarray, config: MultiQTConfig) -> MondrianStream:
    """``multiqt_stream`` run separately on each category's rows (in stream order) from that category's day-one shape."""

    parts = []
    subjects = np.asarray(calibration_subjects).astype(str)
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        own = calibration_groups == group
        if len(np.unique(subjects[own])) < MIN_PATIENTS:
            own = np.ones(len(calibration_scores), dtype=bool)
        parts.append((index, multiqt_stream(
            calibration_scores=calibration_scores[own], calibration_weights=calibration_weights[own],
            predictions=predictions[index], scales=scales[index], observed=observed[index],
            anchor_hours=anchor_hours[index], resolve_hours=resolve_hours[index], config=config)))
    return MondrianStream(rows=len(predictions), parts=tuple(parts))
