"""Anchor-known gating for causal patient residual state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class EventAwareGate:
    """Predict whether applying a resolved patient residual lowers absolute error."""

    model: HistGradientBoostingRegressor
    threshold: float

    def predict_apply(self, features: np.ndarray) -> np.ndarray:
        score = self.model.predict(np.asarray(features, dtype=np.float64))
        return score < self.threshold


def patient_equal_weights(subjects: pd.Series) -> np.ndarray:
    """Give each patient equal total fitting weight."""

    values = subjects.astype(str)
    counts = values.value_counts()
    return values.map(lambda value: 1.0 / counts[value]).to_numpy(dtype=np.float64)


def fit_loss_difference_model(
    features: np.ndarray,
    loss_difference: np.ndarray,
    subjects: pd.Series,
    *,
    seed: int,
) -> HistGradientBoostingRegressor:
    """Fit expected apply-minus-freeze absolute loss from source patients."""

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=160,
        max_leaf_nodes=7,
        min_samples_leaf=25,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=int(seed),
    )
    model.fit(
        np.asarray(features, dtype=np.float64),
        np.asarray(loss_difference, dtype=np.float64),
        sample_weight=patient_equal_weights(subjects),
    )
    return model


def patient_equal_mae(
    subjects: pd.Series,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    error = pd.Series(
        np.abs(np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)),
        index=subjects.astype(str).to_numpy(),
    )
    return float(error.groupby(level=0).mean().mean())


def select_apply_threshold(
    predicted_loss_difference: np.ndarray,
    subjects: pd.Series,
    truth: np.ndarray,
    frozen: np.ndarray,
    always_apply: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    """Select an apply threshold on source tune patients only."""

    score = np.asarray(predicted_loss_difference, dtype=float)
    finite = score[np.isfinite(score)]
    if not len(finite):
        raise ValueError("gate produced no finite tune scores")
    spread = max(float(np.ptp(finite)), 1.0)
    candidates = np.unique(
        np.r_[
            float(np.min(finite) - spread),
            np.quantile(finite, np.linspace(0.05, 0.95, 19)),
            float(np.max(finite) + spread),
        ]
    )
    board = []
    for threshold in candidates:
        apply = score < threshold
        prediction = np.where(apply, always_apply, frozen)
        board.append(
            {
                "threshold": float(threshold),
                "patient_equal_mae": patient_equal_mae(subjects, truth, prediction),
                "apply_fraction": float(apply.mean()),
            }
        )
    board.sort(key=lambda item: (item["patient_equal_mae"], item["apply_fraction"]))
    return float(board[0]["threshold"]), board


def gated_prediction(
    model: HistGradientBoostingRegressor,
    threshold: float,
    features: np.ndarray,
    frozen: np.ndarray,
    always_apply: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = model.predict(np.asarray(features, dtype=np.float64))
    apply = score < float(threshold)
    return np.where(apply, always_apply, frozen), apply, score
