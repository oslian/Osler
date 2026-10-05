"""Constrained, patient-balanced portfolios for near-miss diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor


def unique_prediction_columns(
    predictions: np.ndarray,
    names: Sequence[str],
    *,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, str]]:
    """Remove exact-equivalent prediction sources without reading outcomes."""

    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("prediction matrix and names do not match")
    if not np.isfinite(values).all():
        raise ValueError("predictions must be finite")
    keep: list[int] = []
    aliases: dict[str, str] = {}
    for column, name in enumerate(names):
        duplicate = next(
            (
                prior
                for prior in keep
                if np.max(np.abs(values[:, column] - values[:, prior])) <= tolerance
            ),
            None,
        )
        if duplicate is None:
            keep.append(column)
        else:
            aliases[str(name)] = str(names[duplicate])
    return values[:, keep], tuple(str(names[index]) for index in keep), aliases


def patient_equal_weights(subjects: Sequence[object]) -> np.ndarray:
    """Give each patient equal total objective weight."""

    values = np.asarray(subjects).astype(str)
    if not len(values):
        raise ValueError("subjects cannot be empty")
    names, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if not len(names):
        raise ValueError("subjects cannot be empty")
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def fit_simplex_weights(
    predictions: np.ndarray,
    truth: np.ndarray,
    subjects: Sequence[object],
) -> np.ndarray:
    """Minimize patient-equal absolute error on the probability simplex."""

    matrix = np.asarray(predictions, dtype=float)
    target = np.asarray(truth, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(target):
        raise ValueError("predictions and truth do not match")
    if matrix.shape[1] < 1 or not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("portfolio inputs must be finite and non-empty")
    sample_weight = patient_equal_weights(subjects)

    def objective(weight: np.ndarray) -> float:
        return float(np.sum(sample_weight * np.abs(matrix @ weight - target)))

    initial = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * matrix.shape[1],
        constraints={"type": "eq", "fun": lambda weight: float(weight.sum() - 1.0)},
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"simplex fit failed: {result.message}")
    weight = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    return weight / weight.sum()


def bootstrap_simplex_weights(
    predictions: np.ndarray,
    truth: np.ndarray,
    subjects: Sequence[object],
    seeds: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit patient-bootstrap simplex weights and return their fixed median."""

    matrix = np.asarray(predictions, dtype=float)
    target = np.asarray(truth, dtype=float)
    patient = np.asarray(subjects).astype(str)
    names = np.unique(patient)
    if len(names) < 2:
        raise ValueError("bootstrap portfolio needs at least two patients")
    rows = {name: np.flatnonzero(patient == name) for name in names}
    fitted = []
    for seed in seeds:
        selected = np.random.default_rng(int(seed)).choice(names, len(names), replace=True)
        index = np.concatenate([rows[name] for name in selected])
        sampled_subjects = np.concatenate(
            [np.full(len(rows[name]), f"{draw}:{name}") for draw, name in enumerate(selected)]
        )
        fitted.append(fit_simplex_weights(matrix[index], target[index], sampled_subjects))
    bag_weights = np.asarray(fitted)
    weight = np.median(bag_weights, axis=0)
    if weight.sum() <= 0:
        weight = np.mean(bag_weights, axis=0)
    return weight / weight.sum(), bag_weights


def predict_portfolio(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.asarray(predictions, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or weight.shape != (matrix.shape[1],):
        raise ValueError("portfolio matrix and weights do not match")
    return matrix @ weight


def patient_equal_mae(
    prediction: np.ndarray,
    truth: np.ndarray,
    subjects: Sequence[object],
) -> float:
    error = np.abs(np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float))
    return float(np.sum(patient_equal_weights(subjects) * error))


def state_router_features(
    own_history: np.ndarray,
    predictions: np.ndarray,
    current: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    """Build deployment-legal state and forecast-disagreement features."""

    own = np.asarray(own_history, dtype=float)
    matrix = np.asarray(predictions, dtype=float)
    anchor = np.asarray(current, dtype=float)
    if own.ndim != 2 or matrix.ndim != 2 or len(own) != len(matrix):
        raise ValueError("router features are misaligned")
    if anchor.shape != (len(matrix),) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid router anchor or scale")
    if not np.isfinite(own).all() or not np.isfinite(matrix).all() or not np.isfinite(anchor).all():
        raise ValueError("router features must be finite")
    delta = (matrix - anchor[:, None]) / scale
    return np.column_stack(
        [
            own,
            delta,
            np.abs(delta),
            delta.mean(axis=1),
            delta.std(axis=1),
            np.ptp(delta, axis=1),
        ]
    )


def fit_expected_error_router(
    features: np.ndarray,
    predictions: np.ndarray,
    truth: np.ndarray,
    subjects: Sequence[object],
    *,
    seed: int = 20260908,
) -> tuple[HistGradientBoostingRegressor, ...]:
    """Estimate each fixed source's conditional absolute error."""

    inputs = np.asarray(features, dtype=float)
    matrix = np.asarray(predictions, dtype=float)
    target = np.asarray(truth, dtype=float)
    if inputs.ndim != 2 or matrix.ndim != 2 or len(inputs) != len(matrix):
        raise ValueError("router training arrays are misaligned")
    if target.shape != (len(inputs),):
        raise ValueError("router truth is misaligned")
    weights = patient_equal_weights(subjects)
    options = dict(
        loss="squared_error",
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        learning_rate=0.05,
        early_stopping=False,
        random_state=seed,
    )
    return tuple(
        HistGradientBoostingRegressor(**options).fit(
            inputs, np.abs(matrix[:, column] - target), sample_weight=weights
        )
        for column in range(matrix.shape[1])
    )


def route_expected_error(
    heads: Sequence[HistGradientBoostingRegressor],
    features: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose the fixed source with minimum estimated non-negative error."""

    matrix = np.asarray(predictions, dtype=float)
    if matrix.ndim != 2 or len(heads) != matrix.shape[1]:
        raise ValueError("router heads and predictions do not match")
    expected = np.column_stack(
        [np.maximum(0.0, head.predict(features)) for head in heads]
    )
    choice = np.argmin(expected, axis=1)
    return matrix[np.arange(len(matrix)), choice], choice, expected


__all__ = [
    "bootstrap_simplex_weights",
    "fit_simplex_weights",
    "fit_expected_error_router",
    "patient_equal_mae",
    "patient_equal_weights",
    "predict_portfolio",
    "route_expected_error",
    "state_router_features",
    "unique_prediction_columns",
]
