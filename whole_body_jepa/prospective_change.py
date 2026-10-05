"""Signed-delta readouts and pre-anchor risk calibration for frozen JEPA."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_curve

from .multisource_intervals import conformal_radius
from .probability_calibration import patient_equal_weights, fit_probability_calibrator


def trend_features(frame, rows, target, lookback=24.):
    """Use distinct known measurement times; LOCF is not a new observation."""
    result = []
    groups = frame.groupby("stay_id", sort=False).indices
    for row in rows:
        anchor = frame.iloc[row]
        history = frame.iloc[groups[anchor.stay_id]]
        history = history.loc[history.t <= anchor.t]
        age = pd.to_numeric(history[f"{target}_age_hr"], errors="coerce")
        values = pd.to_numeric(history[f"{target}_t"], errors="coerce")
        times = history.t - pd.to_timedelta(age, unit="h")
        valid = age.ge(0) & np.isfinite(values) & times.notna() & (times <= anchor.t)
        valid &= times >= anchor.t - pd.Timedelta(hours=lookback)
        observations = pd.DataFrame({"time": times[valid], "value": values[valid]}).drop_duplicates("time", keep="last").sort_values("time")
        current = float(anchor[f"{target}_t"])
        slope = span = last_delta = 0.
        if len(observations) >= 2:
            span = (observations.time.iloc[-1] - observations.time.iloc[0]).total_seconds() / 3600
            dt = (observations.time.iloc[-1] - observations.time.iloc[-2]).total_seconds() / 3600
            last_delta = float(observations.value.iloc[-1] - observations.value.iloc[-2])
            slope = last_delta / max(dt, .25)
        result.append([current, slope, float(observations.value.std(ddof=0)) if len(observations) else 0.,
                       float(anchor[f"{target}_age_hr"]), span, len(observations), last_delta])
    return np.asarray(result, dtype=float)


def fit_head(features, delta, subjects, threshold, seed=20260906):
    weights = patient_equal_weights(subjects)
    labels = np.abs(delta) >= threshold
    if min(labels.sum(), (~labels).sum()) < 40:
        raise ValueError("insufficient classes for a change-risk head")
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=40,
                   l2_regularization=1., learning_rate=.05, early_stopping=False, random_state=seed)
    risk = HistGradientBoostingClassifier(**options).fit(features, labels, sample_weight=weights)
    heads = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **options).fit(
        features, delta, sample_weight=weights) for q in (.05, .5, .95)}
    return {"risk": risk, "quantiles": heads}


def raw_prediction(head, features):
    q = np.column_stack([head["quantiles"][q].predict(features) for q in (.05, .5, .95)])
    q.sort(axis=1)
    p = np.clip(head["risk"].predict_proba(features)[:, 1], 1e-6, 1 - 1e-6)
    return {"delta_quantiles": q, "risk_logit": np.log(p / (1 - p))}


def calibrate_risk(score, labels, subjects):
    return fit_probability_calibrator("platt", score, labels, subjects)


def alarm_cutoff(scores, labels, fpr=.05):
    """Strict '>' at a source-only cutoff; ties must not inflate false alerts."""
    negative = np.asarray(scores, dtype=float)[~np.asarray(labels, dtype=bool)]
    if len(negative) < 30 or not np.isfinite(negative).all():
        raise ValueError("insufficient source stable cases")
    return float(np.quantile(negative, 1 - fpr, method="higher"))


def recall_at_fpr(labels, scores, subjects, budget=.05):
    labels = np.asarray(labels, dtype=bool)
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=patient_equal_weights(subjects), drop_intermediate=False)
    return float(tpr[fpr <= budget + 1e-12].max())


def fit_risk_cqr(truth, quantiles, high_risk, min_patients=100):
    """Call on one independent anchor/patient, not repeated correlated scores."""
    truth, quantiles = np.asarray(truth), np.asarray(quantiles)
    errors = np.maximum.reduce([quantiles[:, 0] - truth, truth - quantiles[:, 2], np.zeros(len(truth))])
    return {str(int(high)): conformal_radius(errors[high_risk == high])
            if int(np.sum(high_risk == high)) >= min_patients else None for high in (False, True)}


def cqr_bounds(quantiles, high_risk, radii):
    radius = np.asarray([np.nan if radii[str(int(high))] is None else radii[str(int(high))] for high in high_risk])
    return quantiles[:, 0] - radius, quantiles[:, 2] + radius
