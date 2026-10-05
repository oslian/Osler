"""Measurement-pure non-target history features for representation audits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def matched_wrong_patient_rows(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Pair every row with a time-matched row from a different patient."""

    required = {"subject_id", "_time_hours"}
    if not required.issubset(frame):
        raise ValueError("donor matching requires subject_id and _time_hours")
    subjects = frame["subject_id"].astype(str).to_numpy()
    times = pd.to_numeric(frame["_time_hours"], errors="coerce").to_numpy(float)
    names = np.unique(subjects)
    if len(names) < 2 or not np.isfinite(times).all():
        raise ValueError("donor matching requires two patients and finite times")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(names)
    donor_patient = dict(zip(shuffled, np.roll(shuffled, 1), strict=True))
    patient_rows = {name: np.flatnonzero(subjects == name) for name in names}
    donors = np.empty(len(frame), dtype=np.int64)
    for row, (patient, time) in enumerate(zip(subjects, times, strict=True)):
        candidates = patient_rows[donor_patient[patient]]
        donors[row] = candidates[np.argmin(np.abs(times[candidates] - time))]
    if np.any(subjects[donors] == subjects):
        raise ValueError("wrong-patient donor matched the original patient")
    return donors


def causal_wrong_patient_rows(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Return a wrong-patient state available no later in relative stay time.

    ``-1`` means that the assigned donor has no state yet. This is appropriate
    for prequential placebos: a recipient must not borrow a later point in a
    donor's stay merely because it is the nearest point in absolute distance.
    """

    required = {"subject_id", "_time_hours"}
    if not required.issubset(frame):
        raise ValueError("donor matching requires subject_id and _time_hours")
    subjects = frame["subject_id"].astype(str).to_numpy()
    times = pd.to_numeric(frame["_time_hours"], errors="coerce").to_numpy(float)
    names = np.unique(subjects)
    if len(names) < 2 or not np.isfinite(times).all():
        raise ValueError("donor matching requires two patients and finite times")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(names)
    donor_patient = dict(zip(shuffled, np.roll(shuffled, 1), strict=True))
    patient_rows = {name: np.flatnonzero(subjects == name) for name in names}
    donors = np.full(len(frame), -1, dtype=np.int64)
    for row, (patient, time) in enumerate(zip(subjects, times, strict=True)):
        candidates = patient_rows[donor_patient[patient]]
        available = candidates[times[candidates] <= time]
        if len(available):
            donors[row] = available[np.argmax(times[available])]
    present = donors >= 0
    if np.any(subjects[donors[present]] == subjects[present]):
        raise ValueError("wrong-patient donor matched the original patient")
    if np.any(times[donors[present]] > times[present]):
        raise ValueError("wrong-patient donor used a later relative state")
    return donors


def cross_variable_history_features(
    frame: pd.DataFrame,
    rows: np.ndarray,
    *,
    variables: Sequence[str],
    target: str,
    normalized: np.ndarray,
    observed: np.ndarray,
    ages: np.ndarray,
    history_indices: np.ndarray,
    history_lags_hours: Sequence[float],
    selected_lags_hours: Sequence[float] = (24.0, 12.0, 6.0, 3.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return current and causal history blocks for variables other than target.

    The temporal block stores normalized current-minus-history deltas and a
    validity flag. A delta is valid only when the lag row existed, both values
    were observed, and the underlying measurement timestamps differ. This
    prevents LOCF copies from masquerading as repeated measurements.
    """

    variables = tuple(str(variable) for variable in variables)
    rows = np.asarray(rows, dtype=np.int64)
    history_indices = np.asarray(history_indices, dtype=np.int64)
    lags = np.asarray(tuple(float(value) for value in history_lags_hours), dtype=float)
    selected_lags = tuple(float(value) for value in selected_lags_hours)
    if target not in variables:
        raise ValueError(f"unknown target: {target}")
    if normalized.shape != observed.shape or normalized.shape != ages.shape:
        raise ValueError("normalized, observed, and ages must have equal shape")
    if normalized.shape != (len(frame), len(variables)):
        raise ValueError("observation arrays do not match frame and variables")
    if history_indices.shape != (len(frame), len(lags)):
        raise ValueError("history indices do not match frame and lag contract")
    if np.any(rows < 0) or np.any(rows >= len(frame)):
        raise ValueError("row outside frame")
    if len(set(selected_lags)) != len(selected_lags):
        raise ValueError("selected lags must be unique")

    lag_positions = []
    for lag in selected_lags:
        matches = np.flatnonzero(np.isclose(lags, lag, atol=1e-8))
        if len(matches) != 1:
            raise ValueError(f"selected lag {lag:g}h is absent or ambiguous")
        lag_positions.append(int(matches[0]))

    keep = np.asarray([variable != target for variable in variables], dtype=bool)
    current_value = normalized[rows][:, keep].astype(np.float32)
    current_observed = observed[rows][:, keep] > 0.5
    current_age = ages[rows][:, keep].astype(np.float32)
    current_block = np.concatenate(
        [
            np.where(current_observed, current_value, 0.0),
            current_observed.astype(np.float32),
            np.where(current_observed, current_age, 7.0),
        ],
        axis=1,
    ).astype(np.float32)

    times = pd.to_numeric(frame["_time_hours"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times[rows]).all():
        raise ValueError("anchor times must be finite")
    stay = frame["stay_id"].astype(str).to_numpy()
    anchor_times = times[rows]
    current_measurement_times = anchor_times[:, None] - 24.0 * ages[rows][:, keep]
    temporal_blocks: list[np.ndarray] = []
    for lag, position in zip(selected_lags, lag_positions, strict=True):
        prior_rows = history_indices[rows, position]
        if np.any(stay[prior_rows] != stay[rows]):
            raise ValueError("history crosses patient stays")
        prior_times = times[prior_rows]
        if np.any(prior_times > anchor_times + 1e-8):
            raise ValueError("history contains future rows")
        lag_available = prior_times <= anchor_times - lag + 1e-8
        prior_value = normalized[prior_rows][:, keep]
        prior_observed = observed[prior_rows][:, keep] > 0.5
        prior_measurement_times = prior_times[:, None] - 24.0 * ages[prior_rows][:, keep]
        distinct_measurement = np.abs(
            current_measurement_times - prior_measurement_times
        ) > 1e-6
        valid = (
            current_observed
            & prior_observed
            & lag_available[:, None]
            & distinct_measurement
        )
        delta = np.where(valid, current_value - prior_value, 0.0).astype(np.float32)
        temporal_blocks.extend([delta, valid.astype(np.float32)])

    return current_block, np.concatenate([current_block, *temporal_blocks], axis=1)


def cross_variable_history_views(
    own_history: np.ndarray,
    current_cross: np.ndarray,
    full_cross_history: np.ndarray,
    donor_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build fair current-state and temporal-history attribution views.

    The current-state placebo replaces only the non-target current block.  The
    temporal placebo keeps the patient's own current block and replaces only
    lagged deltas/masks.  This prevents a temporal candidate from winning merely
    because an older placebo also lost the patient's current physiology.
    """

    own_history = np.asarray(own_history, dtype=np.float32)
    current_cross = np.asarray(current_cross, dtype=np.float32)
    full_cross_history = np.asarray(full_cross_history, dtype=np.float32)
    donor_rows = np.asarray(donor_rows, dtype=np.int64)
    rows = len(own_history)
    if own_history.ndim != 2 or current_cross.ndim != 2 or full_cross_history.ndim != 2:
        raise ValueError("history feature views must be matrices")
    if len(current_cross) != rows or len(full_cross_history) != rows or len(donor_rows) != rows:
        raise ValueError("history feature views must have equal row counts")
    if full_cross_history.shape[1] < current_cross.shape[1]:
        raise ValueError("full history cannot be narrower than current cross-state")
    if np.any(donor_rows < 0) or np.any(donor_rows >= rows):
        raise ValueError("donor row outside feature matrix")

    temporal = full_cross_history[:, current_cross.shape[1] :]
    return {
        "own_history": own_history,
        "current_cross": np.column_stack([own_history, current_cross]).astype(np.float32),
        "current_cross_placebo": np.column_stack(
            [own_history, current_cross[donor_rows]]
        ).astype(np.float32),
        "cross_history": np.column_stack(
            [own_history, current_cross, temporal]
        ).astype(np.float32),
        "cross_history_placebo": np.column_stack(
            [own_history, current_cross, temporal[donor_rows]]
        ).astype(np.float32),
    }
