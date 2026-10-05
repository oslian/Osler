"""Choose between fixed predictors using conditional expected absolute error."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


FORECAST_EVIDENCE_NAMES = (
    "anchor_value", "measurement_age_hours", "expert_delta", "expert_q05",
    "expert_q50", "expert_q95", "absolute_expert_delta", "expert_quantile_width",
)


def forecast_evidence(current, age_hours, expert_delta, quantiles, *, scale):
    """Expose only current observations and already-computed expert outputs.

    Quantiles are expert delta estimates, not future observations or validated
    prediction intervals. Physical zero is retained as a measured value.
    """
    current, age_hours, expert_delta, quantiles = (
        np.asarray(x, dtype=float) for x in (current, age_hours, expert_delta, quantiles)
    )
    if (current.ndim != 1 or age_hours.shape != current.shape
            or expert_delta.shape != current.shape or quantiles.shape != (len(current), 3)):
        raise ValueError("forecast evidence arrays are misaligned")
    if not all(np.isfinite(x).all() for x in (current, age_hours, expert_delta, quantiles)):
        raise ValueError("forecast evidence requires finite values")
    if not np.isfinite(scale) or scale <= 0 or np.any(age_hours < 0):
        raise ValueError("invalid scale or measurement age")
    ordered = np.sort(quantiles, axis=1) / scale
    return np.column_stack((current / scale, age_hours, expert_delta / scale,
                            ordered, np.abs(expert_delta) / scale,
                            ordered[:, 2] - ordered[:, 0]))


def fit_cost_router(features, delta, expert_delta, *, seed=20260904):
    """Cost-sensitive expert choice: each label is weighted by absolute regret.

    The resulting probability is a cost-weighted choice score, NOT a calibrated
    probability of patient deterioration. At threshold .5 its population optimum
    chooses the predictor with lower conditional expected absolute error.
    """
    features, delta, expert_delta = (np.asarray(x, dtype=float)
                                    for x in (features, delta, expert_delta))
    if features.ndim != 2 or delta.shape != (len(features),) or expert_delta.shape != delta.shape:
        raise ValueError("cost-routing arrays are misaligned")
    if not all(np.isfinite(x).all() for x in (features, delta, expert_delta)):
        raise ValueError("cost-routing requires finite inputs")
    benefit = np.abs(delta) - np.abs(delta - expert_delta)
    informative = np.abs(benefit) > 0
    labels = (benefit[informative] > 0).astype(int)
    if len(np.unique(labels)) < 2:
        return {"classifier": None, "constant": float(labels[0]) if len(labels) else .5}
    weights = np.abs(benefit[informative])
    weights /= weights.mean()
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=80, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(features[informative], labels, sample_weight=weights)
    return {"classifier": model, "constant": None}


def cost_advantage(head, features):
    model = head["classifier"]
    probability = (np.full(len(features), head["constant"]) if model is None
                   else model.predict_proba(features)[:, 1])
    return 2 * probability - 1


def fit_expected_error_router(features, delta, expert_delta, *, seed=20260904):
    features = np.asarray(features, dtype=float)
    delta = np.asarray(delta, dtype=float)
    expert_delta = np.asarray(expert_delta, dtype=float)
    if features.ndim != 2 or delta.shape != (len(features),) or expert_delta.shape != delta.shape:
        raise ValueError("expected-error training arrays are misaligned")
    if not all(np.isfinite(x).all() for x in (features, delta, expert_delta)):
        raise ValueError("expected-error training requires finite inputs")
    losses = {"persistence": np.abs(delta), "expert": np.abs(delta - expert_delta)}
    losses["benefit"] = losses["persistence"] - losses["expert"]
    heads = {}
    for name, loss in losses.items():
        # Squared error estimates the conditional MEAN of absolute loss.
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=80,
            max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(features, loss)
        heads[name] = model
    return heads


def expected_advantage(heads, features, *, direct=False):
    if direct:
        return heads["benefit"].predict(features)
    persistence = np.maximum(0.0, heads["persistence"].predict(features))
    expert = np.maximum(0.0, heads["expert"].predict(features))
    return persistence - expert


def route_delta(expert_delta, advantage, *, margin=0.0):
    expert_delta = np.asarray(expert_delta, dtype=float)
    advantage = np.asarray(advantage, dtype=float)
    if expert_delta.shape != advantage.shape or not np.isfinite(advantage).all():
        raise ValueError("routing arrays are invalid")
    move = advantage > float(margin)
    return np.where(move, expert_delta, 0.0), move
