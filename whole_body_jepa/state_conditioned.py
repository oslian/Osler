"""State-conditioned factual forecasts with persistence as a stable-state expert."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


DYNAMIC_THRESHOLDS = {
    "creatinine": 0.30,
    "bun": 5.0,
    "urine_output": 50.0,
    "potassium": 0.30,
    "bicarbonate": 2.0,
}
QUANTILES = (0.05, 0.50, 0.95)


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    positive = max(int(labels.sum()), 1)
    negative = max(int((~labels).sum()), 1)
    weights = np.where(labels, len(labels) / (2.0 * positive), len(labels) / (2.0 * negative))
    return weights.astype(np.float64)


def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )


def _regressor(seed: int, *, quantile: float | None = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error" if quantile is None else "quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )


def fit_state_conditioned_head(
    shared_state: np.ndarray,
    target_state: np.ndarray,
    delta: np.ndarray,
    *,
    dynamic_threshold: float,
    seed: int,
) -> dict:
    """Fit P(dynamic), signed dynamic quantiles, and an equal-input plain head."""

    shared_state = np.asarray(shared_state, dtype=np.float64)
    target_state = np.asarray(target_state, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if shared_state.ndim != 2 or target_state.ndim != 2 or len(delta) != len(shared_state):
        raise ValueError("state-conditioned training arrays are misaligned")
    if not np.isfinite(shared_state).all() or not np.isfinite(target_state).all() or not np.isfinite(delta).all():
        raise ValueError("state-conditioned training requires finite arrays")
    dynamic = np.abs(delta) >= float(dynamic_threshold)
    if min(int(dynamic.sum()), int((~dynamic).sum())) < 100:
        raise ValueError("stable and dynamic states each require at least 100 examples")

    classifier = _classifier(seed)
    classifier.fit(shared_state, dynamic.astype(int), sample_weight=_balanced_weights(dynamic))
    quantile_heads = {}
    for offset, quantile in enumerate(QUANTILES):
        model = _regressor(seed + offset + 1, quantile=quantile)
        model.fit(target_state[dynamic], delta[dynamic])
        quantile_heads[str(quantile)] = model
    plain = _regressor(seed + 11)
    plain.fit(target_state, delta)
    return {
        "classifier": classifier,
        "quantile_heads": quantile_heads,
        "plain_head": plain,
        "dynamic_threshold": float(dynamic_threshold),
        "stable_delta_quantiles": np.quantile(delta[~dynamic], np.linspace(0.0, 1.0, 65)),
        "persistence_delta_quantiles": np.quantile(delta, np.linspace(0.0, 1.0, 65)),
        "training_dynamic_fraction": float(dynamic.mean()),
    }


def predict_state_conditioned(
    head: dict,
    shared_state: np.ndarray,
    target_state: np.ndarray,
    *,
    cutoff: float,
) -> dict[str, np.ndarray]:
    probability = head["classifier"].predict_proba(shared_state)[:, 1]
    dynamic_quantiles = np.column_stack(
        [head["quantile_heads"][str(q)].predict(target_state) for q in QUANTILES]
    )
    dynamic_quantiles.sort(axis=1)
    plain_delta = head["plain_head"].predict(target_state)
    # The gate decides whether to move. The signed-delta readout is trained on
    # every outcome, so false-positive dynamic routing is not conditioned on a
    # future-only large-change subset and cannot inflate magnitude by design.
    point_delta = np.where(probability >= cutoff, plain_delta, 0.0)
    return {
        "dynamic_probability": probability,
        "point_delta": point_delta,
        "plain_delta": plain_delta,
        "dynamic_quantiles": dynamic_quantiles,
    }


def mixture_samples(
    probability: np.ndarray,
    dynamic_quantiles: np.ndarray,
    stable_quantiles: np.ndarray,
    *,
    sample_count: int = 64,
) -> np.ndarray:
    """Deterministic samples from stable/dynamic mixtures for proper scoring."""

    probability = np.asarray(probability, dtype=np.float64)
    dynamic_quantiles = np.asarray(dynamic_quantiles, dtype=np.float64)
    stable_quantiles = np.asarray(stable_quantiles, dtype=np.float64)
    u = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
    stable_grid = np.interp(u, np.linspace(0.0, 1.0, len(stable_quantiles)), stable_quantiles)
    output = np.empty((len(probability), sample_count), dtype=np.float64)
    knots = np.asarray((0.0, *QUANTILES, 1.0))
    for row, p_dynamic in enumerate(np.clip(probability, 0.0, 1.0)):
        choose_dynamic = u >= (1.0 - p_dynamic)
        output[row] = stable_grid
        if choose_dynamic.any():
            local = (u[choose_dynamic] - (1.0 - p_dynamic)) / max(p_dynamic, 1e-12)
            values = np.asarray(
                (
                    dynamic_quantiles[row, 0],
                    dynamic_quantiles[row, 0],
                    dynamic_quantiles[row, 1],
                    dynamic_quantiles[row, 2],
                    dynamic_quantiles[row, 2],
                )
            )
            output[row, choose_dynamic] = np.interp(local, knots, values)
    return output


def empirical_crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Exact CRPS of an equally weighted empirical forecast distribution."""

    samples = np.sort(np.asarray(samples, dtype=np.float64), axis=1)
    truth = np.asarray(truth, dtype=np.float64)
    if samples.ndim != 2 or len(truth) != len(samples):
        raise ValueError("CRPS arrays are misaligned")
    first = np.mean(np.abs(samples - truth[:, None]), axis=1)
    m = samples.shape[1]
    weights = 2.0 * np.arange(1, m + 1) - m - 1.0
    pairwise_half = (samples @ weights) / (m * m)
    return first - pairwise_half


def fit_utility_conditioned_head(
    expert_features: np.ndarray,
    expert_delta: np.ndarray,
    router_expert_features: np.ndarray,
    router_state_features: np.ndarray,
    router_delta: np.ndarray,
    *,
    seed: int,
) -> dict:
    """Fit an expert and an out-of-sample router for its realized MAE benefit."""

    expert_features = np.asarray(expert_features, dtype=np.float64)
    expert_delta = np.asarray(expert_delta, dtype=np.float64)
    router_expert_features = np.asarray(router_expert_features, dtype=np.float64)
    router_state_features = np.asarray(router_state_features, dtype=np.float64)
    router_delta = np.asarray(router_delta, dtype=np.float64)
    if expert_features.ndim != 2 or router_expert_features.ndim != 2 or router_state_features.ndim != 2:
        raise ValueError("utility-conditioned features must be matrices")
    if (
        len(expert_features) != len(expert_delta)
        or len(router_expert_features) != len(router_delta)
        or len(router_state_features) != len(router_delta)
    ):
        raise ValueError("utility-conditioned arrays are misaligned")
    if expert_features.shape[1] != router_expert_features.shape[1]:
        raise ValueError("expert feature contracts differ")
    if min(len(expert_delta), len(router_delta)) < 200:
        raise ValueError("utility-conditioned fitting requires at least 200 examples per role")

    point_head = _regressor(seed)
    point_head.fit(expert_features, expert_delta)
    quantile_heads = {}
    for offset, quantile in enumerate(QUANTILES):
        model = _regressor(seed + offset + 1, quantile=quantile)
        model.fit(expert_features, expert_delta)
        quantile_heads[str(quantile)] = model

    router_prediction = point_head.predict(router_expert_features)
    benefit = np.abs(router_delta) - np.abs(router_delta - router_prediction)
    router = fit_utility_router(router_state_features, benefit, seed=seed + 21)
    return {
        "point_head": point_head,
        "quantile_heads": quantile_heads,
        **router,
        "persistence_delta_quantiles": np.quantile(
            np.concatenate([expert_delta, router_delta]), np.linspace(0.0, 1.0, 65)
        ),
    }


def fit_utility_router(
    router_state_features: np.ndarray,
    realized_benefit: np.ndarray,
    *,
    seed: int,
) -> dict:
    """Fit only the shared-state utility router for controlled placebos."""

    router_state_features = np.asarray(router_state_features, dtype=np.float64)
    realized_benefit = np.asarray(realized_benefit, dtype=np.float64)
    if router_state_features.ndim != 2 or len(router_state_features) != len(realized_benefit):
        raise ValueError("utility-router arrays are misaligned")
    beneficial = realized_benefit > 0.0
    if min(int(beneficial.sum()), int((~beneficial).sum())) < 100:
        raise ValueError("beneficial and harmful router outcomes each require at least 100 examples")
    benefit_classifier = _classifier(seed)
    benefit_classifier.fit(
        router_state_features,
        beneficial.astype(int),
        sample_weight=_balanced_weights(beneficial),
    )
    conservative_benefit = _regressor(seed + 1, quantile=0.25)
    conservative_benefit.fit(router_state_features, realized_benefit)
    return {
        "benefit_classifier": benefit_classifier,
        "conservative_benefit_head": conservative_benefit,
        "router_beneficial_fraction": float(beneficial.mean()),
    }


def predict_utility_conditioned(
    head: dict,
    expert_features: np.ndarray,
    router_state_features: np.ndarray,
    *,
    probability_cutoff: float,
) -> dict[str, np.ndarray]:
    """Move only when both benefit probability and its conservative bound agree."""

    expert_features = np.asarray(expert_features, dtype=np.float64)
    router_state_features = np.asarray(router_state_features, dtype=np.float64)
    if len(expert_features) != len(router_state_features):
        raise ValueError("utility-conditioned prediction arrays are misaligned")
    benefit_probability = head["benefit_classifier"].predict_proba(router_state_features)[:, 1]
    conservative_benefit = head["conservative_benefit_head"].predict(router_state_features)
    expert_delta = head["point_head"].predict(expert_features)
    expert_quantiles = np.column_stack(
        [head["quantile_heads"][str(q)].predict(expert_features) for q in QUANTILES]
    )
    expert_quantiles.sort(axis=1)
    move = (benefit_probability >= float(probability_cutoff)) & (conservative_benefit > 0.0)
    return {
        "benefit_probability": benefit_probability,
        "conservative_benefit": conservative_benefit,
        "expert_delta": expert_delta,
        "point_delta": np.where(move, expert_delta, 0.0),
        "expert_quantiles": expert_quantiles,
        "move": move,
    }


def utility_mixture_samples(
    prediction: dict[str, np.ndarray],
    persistence_quantiles: np.ndarray,
    *,
    sample_count: int = 64,
) -> np.ndarray:
    """Build a predictive distribution from persistence and expert utilities."""

    probability = np.where(
        prediction["conservative_benefit"] > 0.0,
        prediction["benefit_probability"],
        0.0,
    )
    return mixture_samples(
        probability,
        prediction["expert_quantiles"],
        np.asarray(persistence_quantiles, dtype=np.float64),
        sample_count=sample_count,
    )


def fit_distribution_expert(
    expert_features: np.ndarray,
    expert_delta: np.ndarray,
    *,
    seed: int,
) -> dict:
    """Fit one all-outcome point head and its conditional delta distribution."""

    expert_features = np.asarray(expert_features, dtype=np.float64)
    expert_delta = np.asarray(expert_delta, dtype=np.float64)
    if expert_features.ndim != 2 or len(expert_features) != len(expert_delta):
        raise ValueError("distribution-expert arrays are misaligned")
    if len(expert_delta) < 200:
        raise ValueError("distribution-expert fitting requires at least 200 examples")
    if not np.isfinite(expert_features).all() or not np.isfinite(expert_delta).all():
        raise ValueError("distribution-expert fitting requires finite arrays")

    point_head = _regressor(seed)
    point_head.fit(expert_features, expert_delta)
    quantile_heads = {}
    for offset, quantile in enumerate(QUANTILES):
        model = _regressor(seed + offset + 1, quantile=quantile)
        model.fit(expert_features, expert_delta)
        quantile_heads[str(quantile)] = model
    return {
        "point_head": point_head,
        "quantile_heads": quantile_heads,
        "persistence_delta_quantiles": np.quantile(
            expert_delta, np.linspace(0.0, 1.0, 65)
        ),
    }


def expert_distribution_samples(
    expert_quantiles: np.ndarray,
    *,
    sample_count: int = 64,
) -> np.ndarray:
    """Interpolate deterministic samples from q05/q50/q95 expert forecasts."""

    expert_quantiles = np.asarray(expert_quantiles, dtype=np.float64)
    if expert_quantiles.ndim != 2 or expert_quantiles.shape[1] != len(QUANTILES):
        raise ValueError("expert quantiles must have shape [rows, 3]")
    ordered = np.sort(expert_quantiles, axis=1)
    u = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
    knots = np.asarray((0.0, *QUANTILES, 1.0), dtype=np.float64)
    output = np.empty((len(ordered), sample_count), dtype=np.float64)
    for row, values in enumerate(ordered):
        padded = np.asarray((values[0], values[0], values[1], values[2], values[2]))
        output[row] = np.interp(u, knots, padded)
    return output


def predict_distribution_risk_conditioned(
    head: dict,
    expert_features: np.ndarray,
    *,
    risk_margin: float,
    sample_count: int = 64,
) -> dict[str, np.ndarray]:
    """Move when the expert distribution assigns lower expected MAE than zero delta."""

    expert_features = np.asarray(expert_features, dtype=np.float64)
    expert_delta = head["point_head"].predict(expert_features)
    expert_quantiles = np.column_stack(
        [head["quantile_heads"][str(q)].predict(expert_features) for q in QUANTILES]
    )
    expert_quantiles.sort(axis=1)
    samples = expert_distribution_samples(expert_quantiles, sample_count=sample_count)
    persistence_risk = np.mean(np.abs(samples), axis=1)
    expert_risk = np.mean(np.abs(samples - expert_delta[:, None]), axis=1)
    risk_advantage = persistence_risk - expert_risk
    relative_risk_advantage = risk_advantage / np.maximum(persistence_risk, 1e-6)
    move = relative_risk_advantage > float(risk_margin)
    return {
        "expert_delta": expert_delta,
        "point_delta": np.where(move, expert_delta, 0.0),
        "expert_quantiles": expert_quantiles,
        "expert_samples": samples,
        "persistence_risk": persistence_risk,
        "expert_risk": expert_risk,
        "risk_advantage": risk_advantage,
        "relative_risk_advantage": relative_risk_advantage,
        "move": move,
    }


def distribution_risk_samples(
    prediction: dict[str, np.ndarray],
    persistence_quantiles: np.ndarray,
) -> np.ndarray:
    """Use the selected expert distribution or the empirical persistence distribution."""

    expert_samples = np.asarray(prediction["expert_samples"], dtype=np.float64)
    persistence_quantiles = np.asarray(persistence_quantiles, dtype=np.float64)
    u = (np.arange(expert_samples.shape[1], dtype=np.float64) + 0.5) / expert_samples.shape[1]
    persistence_samples = np.interp(
        u, np.linspace(0.0, 1.0, len(persistence_quantiles)), persistence_quantiles
    )
    return np.where(prediction["move"][:, None], expert_samples, persistence_samples[None, :])


__all__ = [
    "DYNAMIC_THRESHOLDS",
    "distribution_risk_samples",
    "empirical_crps",
    "expert_distribution_samples",
    "fit_distribution_expert",
    "fit_state_conditioned_head",
    "fit_utility_conditioned_head",
    "fit_utility_router",
    "mixture_samples",
    "predict_state_conditioned",
    "predict_distribution_risk_conditioned",
    "predict_utility_conditioned",
    "utility_mixture_samples",
]
