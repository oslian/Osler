"""Self-calibrating intervals for deployment at a new site without a local calibration study.

Two mature components, applied after a frozen point forecast:

1. Day-one covariate-shift reweighting (weighted split conformal, Tibshirani et
   al. 2019): a domain classifier separates source calibration inputs from the
   new site's *unlabeled* inputs; source calibration scores are reweighted by
   the estimated density ratio before the conformal quantile is taken.
2. Hierarchical online quantile tracking (the P-term of conformal PID control,
   Angelopoulos, Candes and Tibshirani 2023; a variant of adaptive conformal
   inference, Gibbs and Candes 2021): the interval multiplier is tracked in log
   space and updated as each forecast's outcome is actually observed.  Unlike
   re-selecting a quantile of the source scores, the multiplier is unbounded, so
   it can follow a site whose error scale exceeds anything seen in the source.
   The log multiplier is a sum of level terms (site, care unit, value group,
   patient, ...), each keyed by forecast-time information and updated by its
   own step size on the same miscoverage signal.  Level keys are site-local
   (for example the site's own care-unit names), so no cross-site vocabulary
   mapping is needed.

Causality: a forecast's outcome enters the state only once its resolve time has
passed, or once the stream has moved on to a later stay.  Nothing here updates
the point forecast.  Factual research-only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class AdaptiveConfig:
    """Step size per level (0 disables a level) and history needed before a level applies."""

    coverage: float = 0.90
    etas: Mapping[str, float] = field(default_factory=lambda: {"site": 0.05})
    min_history: Mapping[str, int] = field(
        default_factory=lambda: {"site": 0, "care_unit": 20, "value_group": 20, "patient": 3}
    )


def weighted_quantile(scores: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Weighted conformal quantile; the test point's own weight is taken as the mean weight."""

    order = np.argsort(scores, kind="stable")
    sorted_scores = np.asarray(scores, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    total = sorted_weights.sum() + sorted_weights.mean()
    cumulative = np.cumsum(sorted_weights) / total
    index = int(np.searchsorted(cumulative, min(level, 1.0), side="left"))
    if index >= len(sorted_scores):
        return float(sorted_scores[-1])
    return float(sorted_scores[index])


def covariate_shift_weights(
    source: np.ndarray,
    target: np.ndarray,
    *,
    clip_quantile: float = 0.99,
    seed: int = 0,
) -> np.ndarray:
    """Density-ratio weights for source rows from a source-vs-target classifier.

    Only unlabeled target inputs are used.  Weights are truncated at their
    ``clip_quantile`` and normalized to mean one.
    """

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(target) == 0:
        return np.ones(len(source))
    features = np.vstack([source, target])
    labels = np.concatenate([np.zeros(len(source)), np.ones(len(target))])
    classifier = LogisticRegression(C=0.1, max_iter=500, random_state=seed)
    classifier.fit(features, labels)
    probability = np.clip(classifier.predict_proba(source)[:, 1], 1e-4, 1 - 1e-4)
    ratio = probability / (1 - probability) * (len(source) / len(target))
    ratio = np.minimum(ratio, np.quantile(ratio, clip_quantile))
    return ratio / ratio.mean()


@dataclass(frozen=True)
class StreamResult:
    widths: np.ndarray
    log_multiplier: np.ndarray


def adaptive_stream_widths(
    *,
    calibration_scores: np.ndarray,
    calibration_weights: np.ndarray,
    stays: np.ndarray,
    levels: Mapping[str, np.ndarray],
    anchor_hours: np.ndarray,
    resolve_hours: np.ndarray,
    scales: np.ndarray,
    predictions: np.ndarray,
    observed: np.ndarray,
    config: AdaptiveConfig,
) -> StreamResult:
    """Issue widths for one site's stream, rows already in arrival order.

    Width = day-one weighted conformal quantile x row scale x exp(sum of level
    terms).  ``levels`` maps a level name to each row's key at forecast time;
    the ``site`` level needs no keys.  Stays arrive one after another; within a
    stay rows are chronological.  An outcome resolves when ``resolve_hours`` is
    not later than the current anchor of the same stay, or when the stream moves
    to a later stay.
    """

    count = len(stays)
    target = 1.0 - config.coverage
    base = weighted_quantile(calibration_scores, calibration_weights, config.coverage)
    active = {name: eta for name, eta in config.etas.items() if eta > 0}
    keys = {name: (np.zeros(count, dtype=int) if name == "site" else np.asarray(levels[name])) for name in active}
    terms: dict[str, dict[object, float]] = {name: {} for name in active}
    seen: dict[str, dict[object, int]] = {name: {} for name in active}
    pending: list[tuple[float, object, int, float]] = []
    widths = np.full(count, np.nan)
    multipliers = np.zeros(count)

    def resolve(row: int, width: float) -> None:
        if not (np.isfinite(observed[row]) and np.isfinite(predictions[row])):
            return
        step = float(abs(observed[row] - predictions[row]) > width) - target
        for name, eta in active.items():
            key = keys[name][row]
            terms[name][key] = terms[name].get(key, 0.0) + eta * step
            seen[name][key] = seen[name].get(key, 0) + 1

    for row in range(count):
        still_pending = []
        for resolve_time, stay, pending_row, width in pending:
            if resolve_time <= anchor_hours[row]:   # shared deployment clock: resolve only when available
                resolve(pending_row, width)
            else:
                still_pending.append((resolve_time, stay, pending_row, width))
        pending = still_pending

        log_multiplier = 0.0
        for name in active:
            key = keys[name][row]
            if seen[name].get(key, 0) >= config.min_history.get(name, 0):
                log_multiplier += terms[name].get(key, 0.0)
        widths[row] = base * max(float(scales[row]), 1e-9) * float(np.exp(log_multiplier))
        multipliers[row] = log_multiplier
        pending.append((float(resolve_hours[row]), stays[row], row, float(widths[row])))
    return StreamResult(widths=widths, log_multiplier=multipliers)


@dataclass(frozen=True)
class DtaciConfig:
    """Dynamically-tuned tracking (DtACI, Gibbs and Candes 2022) over step-size experts.

    Each expert k tracks its own log multiplier with site step ``eta_k`` and
    care-unit step ``eta_k * unit_ratio``, updated by its own miscoverage.
    Expert weights follow exponential weights on the pinball loss of each
    expert's normalized threshold, with fixed-share mixing so the weights can
    move again after a shift.  The issued threshold is the weighted mean.
    ``initial_log_multiplier`` is every expert's starting site log multiplier
    on the calibration quantile (0 = start from the source calibration).
    """

    coverage: float = 0.90
    expert_etas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
    unit_ratio: float = 0.5
    meta_rate: float = 5.0
    fixed_share: float = 0.005
    min_unit_history: int = 20
    initial_log_multiplier: float = 0.0


def dtaci_stream_widths(
    *,
    calibration_scores: np.ndarray,
    calibration_weights: np.ndarray,
    stays: np.ndarray,
    levels: Mapping[str, np.ndarray],
    anchor_hours: np.ndarray,
    resolve_hours: np.ndarray,
    scales: np.ndarray,
    predictions: np.ndarray,
    observed: np.ndarray,
    config: DtaciConfig,
) -> StreamResult:
    """DtACI version of ``adaptive_stream_widths`` (site + care-unit levels)."""

    count = len(stays)
    target = 1.0 - config.coverage
    base = weighted_quantile(calibration_scores, calibration_weights, config.coverage)
    etas = np.asarray(config.expert_etas, dtype=np.float64)
    experts = len(etas)
    site = np.full(experts, float(config.initial_log_multiplier))
    unit_terms: dict[object, np.ndarray] = {}
    unit_seen: dict[object, int] = {}
    weights = np.full(experts, 1.0 / experts)
    units = np.asarray(levels["care_unit"]) if "care_unit" in levels else np.zeros(count, dtype=int)
    pending: list[tuple[float, int]] = []
    widths = np.full(count, np.nan)
    multipliers = np.zeros(count)

    def thresholds(row: int) -> np.ndarray:
        log_multiplier = site.copy()
        if unit_seen.get(units[row], 0) >= config.min_unit_history:
            log_multiplier = log_multiplier + unit_terms[units[row]]
        return np.exp(log_multiplier)

    def resolve(row: int) -> None:
        # Delayed feedback: the late outcome is compared with each expert's
        # *current* threshold (online quantile tracking, a Robbins-Monro step on
        # the pinball loss), not with the threshold issued when the forecast
        # was made.  With issued thresholds, a burst of stale misses (hundreds
        # of outcomes pending in a network deployment) keeps pushing the shared
        # multiplier up long after it already covers them, and it diverges.
        nonlocal site, weights
        if not (np.isfinite(observed[row]) and np.isfinite(predictions[row])):
            return
        score = abs(observed[row] - predictions[row]) / (base * max(float(scales[row]), 1e-9))
        current = thresholds(row)
        missed = (score > current).astype(float)
        step = missed - target
        site = site + etas * step
        unit = units[row]
        unit_terms[unit] = unit_terms.get(unit, np.zeros(experts)) + etas * config.unit_ratio * step
        unit_seen[unit] = unit_seen.get(unit, 0) + 1
        gap = score - current
        pinball = np.where(gap >= 0, config.coverage * gap, (config.coverage - 1.0) * gap)
        log_weights = np.log(weights) - config.meta_rate * pinball
        updated = np.exp(log_weights - log_weights.max())
        updated /= updated.sum()
        weights = (1.0 - config.fixed_share) * updated + config.fixed_share / experts

    for row in range(count):
        still_pending = []
        for resolve_time, pending_row in pending:
            if resolve_time <= anchor_hours[row]:   # shared deployment clock: resolve only when available
                resolve(pending_row)
            else:
                still_pending.append((resolve_time, pending_row))
        pending = still_pending
        threshold = float(weights @ thresholds(row))
        widths[row] = base * max(float(scales[row]), 1e-9) * threshold
        multipliers[row] = float(np.log(threshold))
        pending.append((float(resolve_hours[row]), row))
    return StreamResult(widths=widths, log_multiplier=multipliers)
