"""Small continual residual adapters with bounded replay memories."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


UPDATE_BATCH_SIZE = 16
ADAPTER_Z_LIMIT = 5.0


class Reservoir:
    """Deterministic fixed-capacity reservoir for feature/label pairs."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        if capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.rng = rng
        self.seen = 0
        self.items: list[tuple[np.ndarray, float]] = []

    def add(self, features: np.ndarray, label: float) -> None:
        item = (np.asarray(features, dtype=np.float64).copy(), float(label))
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        index = int(self.rng.integers(0, self.seen))
        if index < self.capacity:
            self.items[index] = item

    def sample(self, count: int, rng: np.random.Generator) -> list[tuple[np.ndarray, float]]:
        if count <= 0 or not self.items:
            return []
        indices = rng.integers(0, len(self.items), size=int(count))
        return [self.items[int(index)] for index in indices]


def fit_initial_adapter(
    features: np.ndarray,
    residual: np.ndarray,
    *,
    seed: int,
    epochs: int = 30,
) -> tuple[StandardScaler, SGDRegressor, float]:
    """Fit the common initial online adapter on source-only residuals."""

    features = np.asarray(features, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    scaler = StandardScaler().fit(features)
    scale = max(float(np.median(np.abs(residual - np.median(residual))) * 1.4826), 0.05)
    model = SGDRegressor(
        loss="huber",
        epsilon=1.35,
        penalty="l2",
        alpha=1e-3,
        learning_rate="constant",
        eta0=1e-3,
        average=False,
        random_state=int(seed),
    )
    normalized = scaler.transform(features)
    target = residual / scale
    rng = np.random.default_rng(seed)
    for _ in range(int(epochs)):
        order = rng.permutation(len(target))
        model.partial_fit(normalized[order], target[order])
    return scaler, model, scale


def initialize_zero_adapter(
    features: np.ndarray,
    residual_reference: np.ndarray,
    *,
    seed: int,
) -> tuple[StandardScaler, SGDRegressor, float]:
    """Initialize an online adapter whose correction is exactly zero."""

    features = np.asarray(features, dtype=np.float64)
    residual_reference = np.asarray(residual_reference, dtype=np.float64)
    if features.ndim != 2 or not len(features) or len(features) != len(residual_reference):
        raise ValueError("adapter initialization arrays do not align")
    scaler = StandardScaler().fit(features)
    scale = max(
        float(
            np.median(np.abs(residual_reference - np.median(residual_reference)))
            * 1.4826
        ),
        0.05,
    )
    model = SGDRegressor(
        loss="huber",
        epsilon=1.35,
        penalty="l2",
        alpha=1e-3,
        learning_rate="constant",
        eta0=1e-3,
        average=False,
        random_state=int(seed),
    )
    model.partial_fit(scaler.transform(features[:1]), np.zeros(1, dtype=float))
    model.coef_.fill(0.0)
    model.intercept_.fill(0.0)
    return scaler, model, scale


@dataclass
class ContinualArms:
    """Equal-update-budget online residual heads."""

    scaler: StandardScaler
    scale: float
    naive: SGDRegressor
    dual: SGDRegressor
    uniform: SGDRegressor
    short: deque
    long: Reservoir
    uniform_memory: Reservoir
    rng_dual: np.random.Generator
    rng_uniform: np.random.Generator

    @classmethod
    def from_initial(
        cls,
        scaler: StandardScaler,
        model: SGDRegressor,
        scale: float,
        seed_features: np.ndarray,
        seed_labels: np.ndarray,
        *,
        seed: int,
    ) -> "ContinualArms":
        long_rng = np.random.default_rng(seed + 11)
        uniform_rng = np.random.default_rng(seed + 13)
        result = cls(
            scaler=scaler,
            scale=float(scale),
            naive=deepcopy(model),
            dual=deepcopy(model),
            uniform=deepcopy(model),
            short=deque(maxlen=64),
            long=Reservoir(256, long_rng),
            uniform_memory=Reservoir(320, uniform_rng),
            rng_dual=np.random.default_rng(seed + 19),
            rng_uniform=np.random.default_rng(seed + 23),
        )
        for features, label in zip(seed_features, seed_labels, strict=True):
            item = (np.asarray(features, dtype=np.float64).copy(), float(label))
            result.short.append(item)
            result.long.add(*item)
            result.uniform_memory.add(*item)
        return result

    def predict_correction(self, method: str, features: np.ndarray) -> float:
        model = {"naive_online": self.naive, "dual_replay": self.dual,
                 "uniform_replay": self.uniform}[method]
        normalized = self.scaler.transform(np.asarray(features, dtype=float)[None])
        normalized_prediction = float(model.predict(normalized)[0])
        return float(np.clip(normalized_prediction, -ADAPTER_Z_LIMIT, ADAPTER_Z_LIMIT) * self.scale)

    @staticmethod
    def _filled(
        current: tuple[np.ndarray, float],
        sampled: list[tuple[np.ndarray, float]],
        count: int,
    ) -> list[tuple[np.ndarray, float]]:
        return (sampled + [current] * count)[:count]

    def _update_model(
        self,
        model: SGDRegressor,
        batch: list[tuple[np.ndarray, float]],
    ) -> None:
        if len(batch) != UPDATE_BATCH_SIZE:
            raise ValueError("continual update budget mismatch")
        features = self.scaler.transform(np.stack([item[0] for item in batch]))
        labels = np.asarray([item[1] for item in batch], dtype=float) / self.scale
        model.partial_fit(features, labels)

    def update(self, features: np.ndarray, residual: float) -> None:
        current = (np.asarray(features, dtype=np.float64).copy(), float(residual))
        self._update_model(self.naive, [current] * UPDATE_BATCH_SIZE)

        short = list(self.short)
        short_samples = [] if not short else [
            short[int(index)] for index in self.rng_dual.integers(0, len(short), size=6)
        ]
        long_samples = self.long.sample(6, self.rng_dual)
        dual = [current] * 4
        dual += self._filled(current, short_samples, 6)
        dual += self._filled(current, long_samples, 6)
        self._update_model(self.dual, dual)

        historical = self.uniform_memory.sample(12, self.rng_uniform)
        uniform = [current] * 4 + self._filled(current, historical, 12)
        self._update_model(self.uniform, uniform)

        self.short.append(current)
        self.long.add(*current)
        self.uniform_memory.add(*current)


class PatientResidualState:
    """A causal running patient correction using resolved errors only."""

    def __init__(self, prior_strength: float = 3.0) -> None:
        self.prior_strength = float(prior_strength)
        self.sum = 0.0
        self.sum_squares = 0.0
        self.count = 0

    @property
    def correction(self) -> float:
        return self.sum / (self.prior_strength + self.count)

    def update(self, residual: float) -> None:
        self.sum += float(residual)
        self.sum_squares += float(residual) ** 2
        self.count += 1

    def uncertainty_scale(self, reference_scale: float) -> float:
        """Estimate patient error scale with shrinkage to population scale."""

        reference_scale = float(reference_scale)
        if not np.isfinite(reference_scale) or reference_scale <= 0:
            raise ValueError("reference scale must be finite and positive")
        center = self.correction
        squared_error = (
            self.sum_squares - 2.0 * center * self.sum + self.count * center**2
        )
        variance = (
            self.prior_strength * reference_scale**2 + max(squared_error, 0.0)
        ) / (self.prior_strength + self.count)
        return float(np.sqrt(max(variance, 0.05**2)))


class DecayedPatientResidualState:
    """A causal patient residual level with age decay and no trend extrapolation."""

    def __init__(
        self,
        *,
        reference_scale: float,
        half_life_hours: float,
        prior_strength: float,
        max_history: int = 8,
    ) -> None:
        values = (reference_scale, half_life_hours, prior_strength)
        if not all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("decayed state parameters must be finite and positive")
        if max_history < 1:
            raise ValueError("decayed state needs at least one history slot")
        self.reference_scale = float(reference_scale)
        self.half_life_hours = float(half_life_hours)
        self.prior_strength = float(prior_strength)
        self.observations: deque[tuple[datetime, float]] = deque(maxlen=max_history)

    @property
    def count(self) -> int:
        return len(self.observations)

    def update(self, resolved_at: datetime, residual: float) -> None:
        residual = float(residual)
        if not np.isfinite(residual):
            raise ValueError("patient residual must be finite")
        if self.observations and resolved_at < self.observations[-1][0]:
            raise ValueError("patient residuals must resolve in chronological order")
        self.observations.append((resolved_at, residual))

    def estimate(self, anchor: datetime) -> tuple[float, float]:
        """Return a shrunk residual level and its patient-conditioned scale."""

        if not self.observations:
            return 0.0, self.reference_scale
        if any(resolved_at > anchor for resolved_at, _ in self.observations):
            raise ValueError("patient state contains an unresolved future outcome")
        ages = np.asarray(
            [
                (anchor - resolved_at).total_seconds() / 3600.0
                for resolved_at, _ in self.observations
            ],
            dtype=float,
        )
        residuals = np.asarray([value for _, value in self.observations], dtype=float)
        weights = np.exp2(-ages / self.half_life_hours)
        denominator = self.prior_strength + float(weights.sum())
        correction = float(np.sum(weights * residuals) / denominator)
        squared_error = float(np.sum(weights * (residuals - correction) ** 2))
        variance = (
            self.prior_strength * self.reference_scale**2 + squared_error
        ) / denominator
        scale = float(
            np.clip(
                np.sqrt(max(variance, 0.05**2)),
                0.25 * self.reference_scale,
                4.0 * self.reference_scale,
            )
        )
        return correction, scale


class TimeAwarePatientResidualState:
    """Patient-local residual level and trend using only resolved outcomes."""

    def __init__(
        self,
        *,
        reference_scale: float,
        half_life_hours: float,
        prior_strength: float,
        slope_penalty: float,
        max_history: int = 8,
    ) -> None:
        values = (
            reference_scale,
            half_life_hours,
            prior_strength,
            slope_penalty,
        )
        if not all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("time-aware state parameters must be finite and positive")
        if max_history < 2:
            raise ValueError("time-aware state needs at least two history slots")
        self.reference_scale = float(reference_scale)
        self.half_life_hours = float(half_life_hours)
        self.prior_strength = float(prior_strength)
        self.slope_penalty = float(slope_penalty)
        self.observations: deque[tuple[datetime, float]] = deque(maxlen=max_history)

    @property
    def count(self) -> int:
        return len(self.observations)

    def update(self, resolved_at: datetime, residual: float) -> None:
        residual = float(residual)
        if not np.isfinite(residual):
            raise ValueError("patient residual must be finite")
        if self.observations and resolved_at < self.observations[-1][0]:
            raise ValueError("patient residuals must resolve in chronological order")
        self.observations.append((resolved_at, residual))

    def estimate(self, anchor: datetime, horizon_hours: float) -> tuple[float, float]:
        """Return correction and patient-conditioned residual scale."""

        if not self.observations:
            return 0.0, self.reference_scale
        if any(resolved_at > anchor for resolved_at, _ in self.observations):
            raise ValueError("patient state contains an unresolved future outcome")

        due = anchor + timedelta(hours=float(horizon_hours))
        times = np.asarray(
            [
                (resolved_at - due).total_seconds() / 3600.0
                / self.half_life_hours
                for resolved_at, _ in self.observations
            ],
            dtype=float,
        )
        ages = np.asarray(
            [
                (anchor - resolved_at).total_seconds() / 3600.0
                for resolved_at, _ in self.observations
            ],
            dtype=float,
        )
        residuals = np.asarray([value for _, value in self.observations], dtype=float)
        weights = np.exp2(-ages / self.half_life_hours)
        design = np.column_stack([np.ones(len(times)), times])
        penalty = np.diag([self.prior_strength, self.slope_penalty])
        system = design.T @ (weights[:, None] * design) + penalty
        target = design.T @ (weights * residuals)
        intercept, slope = np.linalg.solve(system, target)
        correction = float(
            np.clip(intercept, -2.0 * self.reference_scale, 2.0 * self.reference_scale)
        )
        fitted = design @ np.asarray([intercept, slope])
        residual_variance = float(np.sum(weights * (residuals - fitted) ** 2))
        variance = (
            self.prior_strength * self.reference_scale**2 + residual_variance
        ) / (self.prior_strength + float(weights.sum()))
        scale = float(
            np.clip(
                np.sqrt(max(variance, 0.05**2)),
                0.25 * self.reference_scale,
                4.0 * self.reference_scale,
            )
        )
        return correction, scale
