"""Online hidden-physiology belief proxies.

These features are deliberately *not* direct measurements of hormones,
insulin sensitivity, or buffer capacity.  They are patient-specific,
predict-update proxies inferred from observations available at the anchor.
They may be promoted only when they improve downstream observable forecasts
against both a baseline and a capacity-matched placebo on held-out patients.

The module contains two states requested by the observation layer:

* ``glycemic_clearance_tendency``: an observation-derived proxy for how the
  patient's glucose is clearing, with insulin context used only when it is
  factually observed.  It is not a causal insulin-sensitivity estimate.
* ``acid_base_buffer_reserve``: an observation-derived proxy for remaining
  acid-base buffering/recovery tendency from bicarbonate, pH, anion gap,
  PaCO2, lactate, and their recent trajectory.  It is not a directly measured
  buffer pool.

All state updates are grouped by stay and ordered by an observed time column.
No future-suffixed column is read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GLYCEMIC_CLEARANCE_COLUMNS = (
    "belief_glycemic_clearance_mean",
    "belief_glycemic_clearance_sd",
    "belief_glycemic_clearance_innovation",
    "belief_glycemic_clearance_confidence",
    "belief_glycemic_clearance_observed_insulin",
    "belief_glycemic_clearance_glucose_slope",
)

ACID_BASE_BUFFER_COLUMNS = (
    "belief_acid_base_buffer_reserve_mean",
    "belief_acid_base_buffer_reserve_sd",
    "belief_acid_base_buffer_reserve_innovation",
    "belief_acid_base_buffer_reserve_confidence",
    "belief_acid_base_buffer_stress",
    "belief_acid_base_buffer_observed_components",
)

ALL_HIDDEN_PHYSIOLOGY_COLUMNS = GLYCEMIC_CLEARANCE_COLUMNS + ACID_BASE_BUFFER_COLUMNS


def _num(frame: pd.DataFrame, column: str, default=np.nan) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _clip(values: np.ndarray, low: float, high: float, fill: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out[~np.isfinite(out)] = fill
    return np.clip(out, low, high)


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def _observed_action(frame: pd.DataFrame, stem: str) -> np.ndarray:
    values = np.zeros(len(frame), dtype=np.float64)
    for prefix in ("hist_", "act_"):
        column = f"{prefix}{stem}"
        if column in frame:
            values = np.maximum(values, _clip(_num(frame, column, 0.0), 0.0, 1.0, 0.0))
    return values


def _hours(frame: pd.DataFrame) -> np.ndarray:
    for column in ("hours_since_onset", "hours_since_intime", "t_hour"):
        if column in frame:
            return _clip(_num(frame, column, np.nan), -1.0e6, 1.0e6, 0.0)
    return np.arange(len(frame), dtype=np.float64)


def _ages(frame: pd.DataFrame, columns: tuple[str, ...], default: float = 72.0) -> np.ndarray:
    present = []
    for column in columns:
        if column in frame:
            present.append(_clip(_num(frame, column, default), 0.0, 168.0, default))
    if not present:
        return np.full(len(frame), default, dtype=np.float64)
    return np.nanmin(np.vstack(present), axis=0)


def _stay_values(frame: pd.DataFrame) -> np.ndarray:
    for column in ("stay_id", "patientunitstayid", "subject_id"):
        if column in frame:
            return frame[column].to_numpy()
    return np.arange(len(frame))


def _scalar_state(
    frame: pd.DataFrame,
    observation: np.ndarray,
    confidence: np.ndarray,
    drift_per_hour: np.ndarray,
    *,
    prefix: str,
    low: float,
    high: float,
) -> pd.DataFrame:
    """Run a small scalar predict-update filter in observed time order."""

    names = (
        f"belief_{prefix}_mean",
        f"belief_{prefix}_sd",
        f"belief_{prefix}_innovation",
        f"belief_{prefix}_confidence",
    )
    output = np.full((len(frame), len(names)), np.nan, dtype=np.float64)
    if frame.empty:
        return pd.DataFrame(output, index=frame.index, columns=names)

    observations = _clip(observation, low, high, (low + high) / 2.0)
    confidences = np.clip(np.asarray(confidence, dtype=np.float64), 0.05, 1.0)
    drifts = np.asarray(drift_per_hour, dtype=np.float64).copy()
    drifts[~np.isfinite(drifts)] = 0.0
    work = pd.DataFrame({"_pos": np.arange(len(frame)), "_stay": _stay_values(frame), "_hour": _hours(frame)})

    for _stay, group in work.groupby("_stay", sort=False):
        ordered = group.sort_values("_hour")
        mean = None
        variance = None
        previous_hour = None
        previous_pos = None
        for pos_value, stay_value, hour_value in ordered.itertuples(index=False, name=None):
            pos = int(pos_value)
            obs = float(observations[pos])
            conf = float(confidences[pos])
            obs_var = float(np.clip(0.025 + (1.0 - conf) * 0.45, 0.02, 1.0))
            if mean is None:
                prior_mean = obs
                prior_var = obs_var
                innovation = 0.0
            else:
                delta_h = max(0.0, float(hour_value) - float(previous_hour))
                prior_mean = float(np.clip(mean + drifts[int(previous_pos)] * delta_h, low, high))
                prior_var = float(np.clip(variance + (0.006 + 0.015 * abs(drifts[int(previous_pos)])) * max(delta_h, 0.25), 0.01, 2.0))
                innovation = obs - prior_mean
            posterior_var = 1.0 / (1.0 / max(prior_var, 1e-6) + 1.0 / max(obs_var, 1e-6))
            mean = float(np.clip(posterior_var * (prior_mean / max(prior_var, 1e-6) + obs / max(obs_var, 1e-6)), low, high))
            variance = float(np.clip(posterior_var, 0.006, 2.0))
            output[pos] = (mean, np.sqrt(max(variance, 1e-6)), innovation, conf)
            previous_hour = float(hour_value)
            previous_pos = pos

    return pd.DataFrame(output, index=frame.index, columns=names, dtype=np.float64)


@dataclass(frozen=True)
class GlycemicClearanceBelief:
    """Belief over observed glucose-clearance tendency, not insulin sensitivity."""

    mean: float
    variance: float
    source: str = "observed_glucose_kinetics"

    @property
    def sd(self) -> float:
        return float(np.sqrt(max(self.variance, 1e-6)))

    def predict(self, delta_hours: float, drift: float = 0.0) -> "GlycemicClearanceBelief":
        hours = max(0.0, float(delta_hours))
        return GlycemicClearanceBelief(
            mean=float(np.clip(self.mean + drift * hours, 0.0, 1.0)),
            variance=float(np.clip(self.variance + 0.01 * max(hours, 0.25), 0.01, 2.0)),
            source="predict_update_observed_glucose",
        )

    def update(self, observation: float, confidence: float) -> "GlycemicClearanceBelief":
        obs_var = float(np.clip(0.02 + (1.0 - confidence) * 0.40, 0.02, 1.0))
        post_var = 1.0 / (1.0 / max(self.variance, 1e-6) + 1.0 / obs_var)
        post_mean = post_var * (self.mean / max(self.variance, 1e-6) + observation / obs_var)
        return GlycemicClearanceBelief(float(np.clip(post_mean, 0.0, 1.0)), float(np.clip(post_var, 0.006, 2.0)))


@dataclass(frozen=True)
class AcidBaseBufferBelief:
    """Belief over a measured-component acid-base reserve proxy."""

    mean: float
    variance: float
    source: str = "observed_acid_base_components"

    @property
    def sd(self) -> float:
        return float(np.sqrt(max(self.variance, 1e-6)))

    def predict(self, delta_hours: float, drift: float = 0.0) -> "AcidBaseBufferBelief":
        hours = max(0.0, float(delta_hours))
        return AcidBaseBufferBelief(
            mean=float(np.clip(self.mean + drift * hours, 0.0, 1.0)),
            variance=float(np.clip(self.variance + 0.012 * max(hours, 0.25), 0.01, 2.0)),
            source="predict_update_observed_acid_base",
        )

    def update(self, observation: float, confidence: float) -> "AcidBaseBufferBelief":
        obs_var = float(np.clip(0.025 + (1.0 - confidence) * 0.45, 0.02, 1.0))
        post_var = 1.0 / (1.0 / max(self.variance, 1e-6) + 1.0 / obs_var)
        post_mean = post_var * (self.mean / max(self.variance, 1e-6) + observation / obs_var)
        return AcidBaseBufferBelief(float(np.clip(post_mean, 0.0, 1.0)), float(np.clip(post_var, 0.006, 2.0)))


def hidden_physiology_belief_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return online, future-safe hidden-state proxy features for ``frame``."""

    glucose = _clip(_num(frame, "glucose_t"), 20.0, 900.0, 140.0)
    glucose_age = _ages(frame, ("glucose_age_hr",))
    bicarbonate = _clip(_num(frame, "bicarbonate_t"), 3.0, 50.0, 24.0)
    ph = _clip(_num(frame, "ph_t"), 6.5, 8.0, 7.4)
    anion_gap = _clip(_num(frame, "anion_gap_t"), 0.0, 70.0, 12.0)
    paco2 = _clip(_num(frame, "paco2_t"), 5.0, 200.0, 40.0)
    lactate = _clip(_num(frame, "lactate_t"), 0.0, 30.0, 1.5)
    insulin = _observed_action(frame, "insulin")
    dextrose = _observed_action(frame, "dextrose")

    # Clearance is inferred from observed glucose kinetics and factual insulin
    # context.  When insulin is absent, confidence is intentionally low.
    glucose_slope = np.zeros(len(frame), dtype=np.float64)
    if len(frame):
        work = pd.DataFrame({"_stay": _stay_values(frame), "_hour": _hours(frame), "_glucose": glucose})
        for _stay, group in work.groupby("_stay", sort=False):
            ordered = group.sort_values("_hour")
            previous = None
            previous_hour = None
            for row_index, _stay_value, hour_value, glucose_value in ordered.itertuples(index=True, name=None):
                if previous is not None:
                    dt = max(0.25, float(hour_value) - float(previous_hour))
                    glucose_slope[int(row_index)] = np.clip((float(glucose_value) - previous) / dt, -200.0, 200.0)
                previous = float(glucose_value)
                previous_hour = float(hour_value)
    fall_signal = _clip01((-glucose_slope - 0.01) / 0.20)
    rise_signal = _clip01((glucose_slope - 0.01) / 0.20)
    clearance_obs = _clip01(0.50 * fall_signal + 0.20 * (1.0 - rise_signal) + 0.20 * insulin + 0.10 * (1.0 - dextrose))
    clearance_conf = _clip01((0.20 + 0.80 * insulin) * (1.0 - 0.75 * glucose_age / 168.0))
    clearance_drift = -0.004 * insulin + 0.003 * dextrose
    clearance_state = _scalar_state(
        frame, clearance_obs, clearance_conf, clearance_drift,
        prefix="glycemic_clearance", low=0.0, high=1.0,
    )

    # Reserve is high when bicarbonate/pH are preserved and gap/lactate/PaCO2
    # do not indicate a major acid load.  These are observed-component proxies.
    bicarb_support = _clip01((bicarbonate - 12.0) / 16.0)
    ph_support = _clip01((ph - 7.05) / 0.35)
    gap_burden = _clip01((anion_gap - 12.0) / 28.0)
    lactate_burden = _clip01((lactate - 2.0) / 8.0)
    paco2_burden = _clip01(np.abs(paco2 - 40.0) / 45.0)
    buffer_stress = _clip01(0.36 * (1.0 - bicarb_support) + 0.24 * (1.0 - ph_support) + 0.20 * gap_burden + 0.12 * lactate_burden + 0.08 * paco2_burden)
    buffer_obs = _clip01(1.0 - buffer_stress)
    component_ages = _ages(frame, ("bicarbonate_age_hr", "ph_age_hr", "anion_gap_age_hr", "lactate_age_hr", "paco2_age_hr"))
    component_count = sum(f"{column}_t" in frame for column in ("bicarbonate", "ph", "anion_gap", "lactate", "paco2"))
    buffer_conf = _clip01((0.10 + 0.18 * component_count) * (1.0 - 0.80 * component_ages / 168.0))
    buffer_drift = 0.004 * (1.0 - buffer_stress) - 0.004 * insulin
    buffer_state = _scalar_state(
        frame, buffer_obs, buffer_conf, buffer_drift,
        prefix="acid_base_buffer_reserve", low=0.0, high=1.0,
    )

    output = pd.concat(
        [
            clearance_state,
            pd.DataFrame({
                "belief_glycemic_clearance_observed_insulin": insulin,
                "belief_glycemic_clearance_glucose_slope": glucose_slope,
            }, index=frame.index),
            buffer_state,
            pd.DataFrame({
                "belief_acid_base_buffer_stress": buffer_stress,
                "belief_acid_base_buffer_observed_components": float(component_count),
            }, index=frame.index),
        ],
        axis=1,
    )
    return output.reindex(columns=ALL_HIDDEN_PHYSIOLOGY_COLUMNS).replace([np.inf, -np.inf], np.nan).astype(np.float64)


def placebo_hidden_physiology_features(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Capacity-matched placebo features for the hidden-state audit."""

    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(frame), len(ALL_HIDDEN_PHYSIOLOGY_COLUMNS)))
    return pd.DataFrame(
        values,
        index=frame.index,
        columns=[f"placebo_{column}" for column in ALL_HIDDEN_PHYSIOLOGY_COLUMNS],
        dtype=np.float64,
    )
