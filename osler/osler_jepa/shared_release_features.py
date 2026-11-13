"""Canonical feature assembly for the shared whole-body JEPA release.

Serving code must never construct the 311-value target-head vector by hand.
The block order is part of the trained model and a misplaced scalar changes
the meaning of every downstream feature without producing a shape error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SHARED_CONTEXT_WIDTH = 64
BELIEF_WIDTH = 96
TARGET_FUTURE_WIDTH = 64
ORGAN_WIDTH = 32
MEASUREMENT_STATE_WIDTH = 53
FEATURE_WIDTH = 311

FEATURE_SLICES = {
    "shared_context": slice(0, 64),
    "patient_belief": slice(64, 160),
    "target_future": slice(160, 224),
    "target_organ": slice(224, 256),
    "base_point": slice(256, 257),
    "measurement_pure_state": slice(257, 310),
    "target_current": slice(310, 311),
}


def _finite_vector(name: str, value: Sequence[float], width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (width,):
        raise ValueError(f"{name} must have width {width}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class MeasurementState:
    """Normalized anchor state plus an audit-only observed mask."""

    normalized: np.ndarray
    observed: np.ndarray


def normalize_measurement_state(
    values: Mapping[str, float | None],
    *,
    variables: Sequence[str],
    medians: Sequence[float],
    scales: Sequence[float],
) -> MeasurementState:
    """Normalize the latest valid pre-anchor value for every variable.

    Missing values become the training median (zero in normalized space).
    A physical value of zero remains an observed zero and is normalized in the
    same way as every other value. It is never used as a missing-value marker.
    """

    variables = tuple(str(variable) for variable in variables)
    if len(variables) != MEASUREMENT_STATE_WIDTH:
        raise ValueError(
            f"release requires {MEASUREMENT_STATE_WIDTH} variables, got {len(variables)}"
        )
    median = _finite_vector("medians", medians, len(variables))
    scale = _finite_vector("scales", scales, len(variables))
    if np.any(scale <= 0.0):
        raise ValueError("scales must be positive")

    raw = np.empty(len(variables), dtype=np.float32)
    observed = np.zeros(len(variables), dtype=bool)
    for index, variable in enumerate(variables):
        value = values.get(variable)
        if value is None:
            raw[index] = median[index]
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            raw[index] = median[index]
            continue
        raw[index] = numeric
        observed[index] = True
    return MeasurementState(
        normalized=((raw - median) / scale).astype(np.float32),
        observed=observed,
    )


def build_target_head_features(
    *,
    shared_context: Sequence[float],
    patient_belief: Sequence[float],
    target_future: Sequence[float],
    target_organ: Sequence[float],
    base_forecast_normalized: float,
    measurement_pure_state: Sequence[float],
    target_current_normalized: float,
) -> np.ndarray:
    """Build the exact feature vector expected by every release target head.

    ``base_forecast_normalized`` is the frozen shared JEPA numerical forecast
    for this target and horizon. It is not the anchor target value. The anchor
    value is supplied separately as ``target_current_normalized``.
    """

    base = float(base_forecast_normalized)
    current = float(target_current_normalized)
    if not np.isfinite(base) or not np.isfinite(current):
        raise ValueError("base forecast and current target must be finite")
    result = np.concatenate(
        [
            _finite_vector("shared_context", shared_context, SHARED_CONTEXT_WIDTH),
            _finite_vector("patient_belief", patient_belief, BELIEF_WIDTH),
            _finite_vector("target_future", target_future, TARGET_FUTURE_WIDTH),
            _finite_vector("target_organ", target_organ, ORGAN_WIDTH),
            np.asarray([base], dtype=np.float32),
            _finite_vector(
                "measurement_pure_state",
                measurement_pure_state,
                MEASUREMENT_STATE_WIDTH,
            ),
            np.asarray([current], dtype=np.float32),
        ]
    )
    if result.shape != (FEATURE_WIDTH,):
        raise AssertionError(f"feature contract produced {result.shape}, expected 311")
    return result


__all__ = [
    "FEATURE_SLICES",
    "FEATURE_WIDTH",
    "MeasurementState",
    "build_target_head_features",
    "normalize_measurement_state",
]
