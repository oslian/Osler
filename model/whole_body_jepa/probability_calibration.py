"""Patient-weighted probability calibration for factual event-risk heads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def patient_equal_weights(subjects: np.ndarray) -> np.ndarray:
    values = np.asarray(subjects).astype(str)
    unique, counts = np.unique(values, return_counts=True)
    count = dict(zip(unique, counts))
    weights = np.asarray([1.0 / count[value] for value in values], dtype=float)
    return weights * (len(weights) / weights.sum())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


@dataclass
class ProbabilityCalibrator:
    method: str
    model: object

    def predict(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=float).reshape(-1)
        if self.method == "temperature":
            return _sigmoid(values / float(self.model))
        if self.method == "platt":
            return self.model.predict_proba(values[:, None])[:, 1]
        if self.method == "beta":
            probability = np.clip(_sigmoid(values), 1e-6, 1.0 - 1e-6)
            features = np.column_stack((np.log(probability), -np.log1p(-probability)))
            return self.model.predict_proba(features)[:, 1]
        if self.method == "isotonic":
            return np.asarray(self.model.predict(values), dtype=float)
        raise ValueError(f"unknown calibration method: {self.method}")


def fit_probability_calibrator(
    method: str,
    logits: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> ProbabilityCalibrator:
    values = np.asarray(logits, dtype=float).reshape(-1)
    truth = np.asarray(labels, dtype=int).reshape(-1)
    weights = patient_equal_weights(subjects)
    if len(np.unique(truth)) < 2:
        raise ValueError("probability calibration requires both event classes")
    if method == "temperature":
        grid = np.geomspace(0.35, 4.0, 100)
        losses = []
        for temperature in grid:
            probability = np.clip(_sigmoid(values / temperature), 1e-7, 1.0 - 1e-7)
            loss = -(
                truth * np.log(probability) + (1 - truth) * np.log1p(-probability)
            )
            losses.append(float(np.average(loss, weights=weights)))
        return ProbabilityCalibrator(method, float(grid[int(np.argmin(losses))]))
    if method == "platt":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(values[:, None], truth, sample_weight=weights)
        return ProbabilityCalibrator(method, model)
    if method == "beta":
        probability = np.clip(_sigmoid(values), 1e-6, 1.0 - 1e-6)
        features = np.column_stack((np.log(probability), -np.log1p(-probability)))
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(features, truth, sample_weight=weights)
        return ProbabilityCalibrator(method, model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5)
        model.fit(values, truth, sample_weight=weights)
        return ProbabilityCalibrator(method, model)
    raise ValueError(f"unknown calibration method: {method}")
