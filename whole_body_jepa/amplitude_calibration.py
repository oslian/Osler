"""Learn bounded adoption of a frozen expert's delta under absolute error."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def fit_amplitude(features, delta, expert_delta, *, conditional=True, seed=20260904):
    features, delta, expert_delta = (np.asarray(x, dtype=float)
                                    for x in (features, delta, expert_delta))
    if (features.ndim != 2 or delta.shape != (len(features),)
            or expert_delta.shape != delta.shape or not len(features)):
        raise ValueError("amplitude training arrays are misaligned or empty")
    if not all(np.isfinite(x).all() for x in (features, delta, expert_delta)):
        raise ValueError("amplitude training requires finite inputs")
    use = expert_delta != 0
    if not use.any():
        return {"model": None, "constant": 0.0}
    d, truth = expert_delta[use], delta[use]
    # |truth - a*d| = |d|*|truth/d - a|. Clipping truth/d to [0,1]
    # changes this objective only by a constant for bounded adoption a.
    ratio = np.clip(np.clip(truth, -np.abs(d), np.abs(d)) / d, 0, 1)
    weights = np.abs(d)
    weights /= weights.max()
    if not conditional or np.ptp(ratio) == 0:
        order = np.argsort(ratio, kind="stable")
        index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)
        return {"model": None, "constant": float(ratio[order[index]])}
    model = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=.05, max_iter=80,
        max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=False, random_state=seed,
    )
    model.fit(features[use], ratio, sample_weight=weights / weights.mean())
    return {"model": model, "constant": None}


def adoption(head, features):
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("invalid amplitude inputs")
    value = (np.full(len(features), head["constant"]) if head["model"] is None
             else head["model"].predict(features))
    return np.clip(value, 0, 1)
