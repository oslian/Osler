"""Diagnostic stable/change/uncertainty scoring, never a routing policy."""

import numpy as np
import pandas as pd

from .state_conditioned import DYNAMIC_THRESHOLDS


# Research effect sizes in canonical physical units, not clinical alert limits.
CHANGE_THRESHOLDS = {
    **DYNAMIC_THRESHOLDS, "glucose": 20., "heart_rate": 10., "map": 10.,
    "respiratory_rate": 3., "o2sat": 3., "hematocrit": 3.,
    "hemoglobin": 1., "platelets": 20., "sodium": 3., "chloride": 3.,
}


def estimate(frame, values, *, seed=20260906, repetitions=1000):
    """Equal patient weight, then paired patient bootstrap (including repeats)."""
    values = np.asarray(values, dtype=float)
    if len(values) != len(frame):
        raise ValueError("metric and patient rows do not align")
    finite = np.isfinite(values)
    means = pd.Series(values[finite], index=frame.loc[finite, "subject_id"].to_numpy()).groupby(level=0).mean().to_numpy()
    if not len(means):
        return {"mean": None, "ci95": None, "patients": 0, "supported": False}
    ci = None
    if len(means) >= 30:
        rng = np.random.default_rng(seed)
        samples = np.concatenate([rng.choice(means, (min(100, repetitions - k), len(means))).mean(axis=1)
                                  for k in range(0, repetitions, 100)])
        ci = list(map(float, np.quantile(samples, [.025, .975])))
    return {"mean": float(means.mean()), "ci95": ci, "patients": len(means), "supported": len(means) >= 30}


def paired_model_errors(frame, predictions, reference, *, seed, repetitions=20000):
    """Common complete rows and patient draws, independent of model names."""
    truth = frame.truth.to_numpy(float)
    if reference not in predictions:
        raise ValueError('reference prediction required')
    arrays = {name: np.asarray(value, dtype=float) for name, value in predictions.items()}
    if any(value.shape != truth.shape for value in arrays.values()):
        raise ValueError('predictions must align with truth')
    if frame.subject_id.isna().any() or not np.isfinite(truth).all() or any(
            not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('common finite predictions and patient IDs required')
    if repetitions < 1:
        raise ValueError('positive bootstrap repetitions required')
    errors = {name: np.abs(value - truth) for name, value in arrays.items()}
    return {name: estimate(frame, error - errors[reference], seed=seed, repetitions=repetitions)
            for name, error in errors.items() if name != reference}


def interval_score(truth, lower, upper, alpha=.1):
    """Central interval score: width plus misses; lower is better, not CRPS."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (truth, lower, upper))
    if not 0 < alpha < 1 or np.any(lo > hi):
        raise ValueError("invalid interval")
    return hi - lo + 2 / alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def _regime(frame, threshold, repetitions):
    frame = frame.reset_index(drop=True)
    def metric(v):
        return estimate(frame, v, repetitions=repetitions)
    y, x, pred, lo, hi = (frame[c].to_numpy(float) for c in ("truth", "current", "point", "lower", "upper"))
    predicted = np.isfinite(pred)
    intervals = predicted & np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
    delta, pdelt = y - x, pred - x
    actual_change = np.abs(delta) >= threshold
    move = np.abs(pdelt) >= threshold
    error = np.abs(pred - y)
    hit = (lo <= y) & (y <= hi)
    def where(mask, v):
        return metric(np.where(mask, v, np.nan))
    scores = interval_score(y, np.where(intervals, lo, np.nan), np.where(intervals, hi, np.nan))
    result = {
        "rows": len(frame), "patients": int(frame.subject_id.nunique()),
        "prediction_availability": metric(predicted),
        "abstention_fraction": metric(~predicted),
        "interval_availability": metric(intervals),
        "mae": where(predicted, error),
        "persistence_mae_same_rows": where(predicted, np.abs(delta)),
        "mae_delta_vs_persistence": where(predicted, error - np.abs(delta)),
        "signed_delta_bias": where(predicted, pdelt - delta),
        "magnitude_error": where(predicted, np.abs(np.abs(pdelt) - np.abs(delta))),
        "large_move_fraction": where(predicted, move),
        "false_move_rate_on_stable": where(predicted & ~actual_change, move),
        "change_recall_including_abstention": where(actual_change, predicted & move),
        "signed_change_recall_including_abstention": where(actual_change, predicted & move & (np.sign(pdelt) == np.sign(delta))),
        "miss_or_abstain_rate_on_change": where(actual_change, ~(predicted & move)),
        "direction_accuracy_on_change": where(predicted & actual_change, np.sign(pdelt) == np.sign(delta)),
        "coverage_when_interval_available": where(intervals, hit),
        "covered_fraction_all_evaluable": metric(intervals & hit),
        "interval_width": where(intervals, hi - lo),
        "interval_score_90": where(intervals, scores),
    }
    cov = result["coverage_when_interval_available"]
    result["coverage_status"] = ("insufficient_support" if not cov["supported"] else
                                  "within_nominal_band" if .87 <= cov["mean"] <= .93 else "outside_nominal_band")
    result["selective_intervals"] = {}
    # Selection depends only on issued widths, not on whether the outcome hit.
    for name, mask in (("narrow", intervals & ((hi - lo) <= 2 * threshold)),
                       ("wide", intervals & ((hi - lo) > 2 * threshold))):
        result["selective_intervals"][name] = {
            "fraction": metric(mask), "coverage": where(mask, hit), "mae": where(mask, error)}
    return result


def evaluate_change(frame, threshold, *, repetitions=1000):
    """Future values define retrospective strata ONLY; all predictions stay frozen."""
    required = {"subject_id", "current", "truth", "point", "lower", "upper"}
    if not required.issubset(frame):
        raise ValueError(f"missing scored columns: {sorted(required - set(frame))}")
    if not np.isfinite(threshold) or threshold <= 0 or repetitions < 1:
        raise ValueError("positive threshold and bootstrap count required")
    if frame.subject_id.isna().any():
        raise ValueError("missing patient identity")
    valid = np.isfinite(frame.current) & np.isfinite(frame.truth)
    frame = frame.loc[valid].reset_index(drop=True)
    both = np.isfinite(frame.lower) & np.isfinite(frame.upper)
    if (frame.loc[both, "lower"] > frame.loc[both, "upper"]).any():
        raise ValueError("reversed interval")
    if (both & ~np.isfinite(frame.point)).any():
        raise ValueError("interval cannot accompany abstained point")
    change = (frame.truth - frame.current).abs() >= threshold
    return {
        "change_threshold": float(threshold), "threshold_is_clinical": False,
        "future_strata_used_for_routing": False,
        "all": _regime(frame, threshold, repetitions),
        "stable": _regime(frame.loc[~change], threshold, repetitions),
        "changed": _regime(frame.loc[change], threshold, repetitions),
        "promotion_allowed": False,
    }
