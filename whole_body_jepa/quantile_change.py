"""Quantile-distribution diagnostics on a frozen shared-state representation."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .multisource_intervals import conformal_radius
from .probability_calibration import patient_equal_weights
from .prospective_change import recall_at_fpr


QUANTILES = np.arange(1, 20, dtype=float) / 20


def fit_distribution(features, delta, subjects):
    weights = patient_equal_weights(subjects)
    return [HistGradientBoostingRegressor(loss="quantile", quantile=float(q),
        max_iter=80, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.,
        learning_rate=.05, early_stopping=False, random_state=20260906).fit(
            features, delta, sample_weight=weights) for q in QUANTILES]


def predict_distribution(heads, features):
    return np.sort(np.column_stack([head.predict(features) for head in heads]), axis=1)


def tail_probabilities(quantiles, threshold):
    """Integrate a piecewise-linear quantile function, including endpoint atoms."""
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 2 or q.shape[1] != 19 or not np.isfinite(q).all() or np.any(np.diff(q, axis=1) < 0):
        raise ValueError("requires 19 finite ordered quantiles")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("positive change threshold required")
    values = np.column_stack([q[:, 0], q, q[:, -1]])
    weights = np.diff(np.r_[0., QUANTILES, 1.])
    left, right = values[:, :-1], values[:, 1:]
    span = right - left

    def cdf(value, inclusive):
        fraction = np.clip(np.divide(value - left, span, out=np.zeros_like(span), where=span > 0), 0, 1)
        atoms = left <= value if inclusive else left < value
        fraction = np.where(span == 0, atoms, fraction)
        return np.sum(fraction * weights, axis=1)

    down = cdf(-threshold, True)
    up = 1 - cdf(threshold, False)
    return np.column_stack([down, np.maximum(0., 1 - down - up), up])


def pinball_score(truth, quantiles):
    error = np.asarray(truth)[:, None] - quantiles
    return np.maximum(QUANTILES * error, (QUANTILES - 1) * error).mean(axis=1)


def interval_scale(quantiles, threshold):
    return np.maximum(quantiles[:, -1] - quantiles[:, 0], threshold)


def fit_pooled_cqr(truth, quantiles, subjects, threshold, normalized):
    subjects = np.asarray(subjects).astype(str)
    if len(np.unique(subjects)) != len(subjects):
        raise ValueError("CQR requires one calibration score per patient")
    if len(subjects) < 100:
        return None
    errors = np.maximum.reduce([quantiles[:, 0] - truth, truth - quantiles[:, -1], np.zeros(len(truth))])
    scale = interval_scale(quantiles, threshold) if normalized else np.ones(len(truth))
    return conformal_radius(errors / scale)


def pooled_bounds(quantiles, threshold, radius, normalized):
    if radius is None:
        return np.full(len(quantiles), np.nan), np.full(len(quantiles), np.nan)
    scale = interval_scale(quantiles, threshold) if normalized else np.ones(len(quantiles))
    return quantiles[:, 0] - radius * scale, quantiles[:, -1] + radius * scale


def paired_ranking_recall(labels, candidate, baseline, seed=20260907, repetitions=1000):
    """One row per patient; held-out ROC cutoffs are diagnostics, never serving."""
    labels = np.asarray(labels, dtype=bool)
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    if min(labels.sum(), (~labels).sum()) < 30:
        return {"mean": None, "ci95": None, "reason": "insufficient_classes"}
    subjects = np.arange(len(labels))
    difference = recall_at_fpr(labels, candidate, subjects) - recall_at_fpr(labels, baseline, subjects)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        ix = rng.integers(len(labels), size=len(labels))
        y = labels[ix]
        if len(np.unique(y)) < 2:
            continue
        # Each bootstrap draw is an independent draw from the patient empirical measure.
        draws.append(recall_at_fpr(y, candidate[ix], subjects) - recall_at_fpr(y, baseline[ix], subjects))
    return {"mean": float(difference), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "patients": len(labels), "ranking_diagnostic_only": True}
