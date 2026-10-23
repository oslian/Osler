"""Auditable beat-level encoders for MIMIC-IV waveform candidates.

This module turns bounded ECG/ABP/PLETH/RESP windows into physiologic cycle
features.  It is deliberately a feature encoder, not a clinical monitor:
all outputs remain candidate-only until the downstream held-out audit passes.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from osler_jepa.waveform import classify_signal_name


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _filled(values: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values), 0.0
    median = float(np.median(values[finite]))
    return np.where(finite, values, median), float(finite.mean())


def _filter(values: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """Apply a conservative band-pass when the window is long enough."""

    filled, _ = _filled(values)
    if len(filled) < 32 or not np.isfinite(fs) or fs <= 2.0:
        return filled - float(np.median(filled))
    nyquist = fs / 2.0
    low = max(0.01, min(float(low), nyquist * 0.45))
    high = max(low + 0.01, min(float(high), nyquist * 0.9))
    if high >= nyquist:
        return filled - float(np.median(filled))
    try:
        sos = butter(3, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        return sosfiltfilt(sos, filled)  # type: ignore[no-any-return]
    except ValueError:
        return filled - float(np.median(filled))


def _quality(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    valid_fraction = float(finite.mean()) if len(values) else 0.0
    clean = values[finite]
    if len(clean) < 2:
        flatline = 1.0
    else:
        scale = max(float(np.nanstd(clean)), 1e-9)
        flatline = float(np.mean(np.abs(np.diff(clean)) <= scale * 1e-4))
    return {
        "valid_fraction": valid_fraction,
        "flatline_fraction": flatline,
        "finite_sample_count": float(finite.sum()),
    }


def _peaks(
    values: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    min_interval_s: float,
    prominence_floor: float = 0.35,
) -> np.ndarray:
    filtered = _filter(values, fs, low_hz, high_hz)
    scale = float(np.nanstd(filtered))
    if scale <= 1e-9:
        return np.asarray([], dtype=int)
    normalized = filtered / scale
    distance = max(1, int(round(float(min_interval_s) * fs)))
    prominence = max(
        float(prominence_floor),
        float(np.nanpercentile(np.abs(normalized), 60)) * 0.25,
    )
    peaks, _ = find_peaks(normalized, distance=distance, prominence=prominence)
    return peaks.astype(int)


def _interval_stats(peaks: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    output = {
        f"{prefix}_beat_count": float(len(peaks)),
        f"{prefix}_rate_per_min": math.nan,
        f"{prefix}_interval_median_s": math.nan,
        f"{prefix}_interval_iqr_s": math.nan,
        f"{prefix}_sdnn_s": math.nan,
        f"{prefix}_rmssd_s": math.nan,
    }
    if len(peaks) < 2 or fs <= 0:
        return output
    intervals = np.diff(peaks).astype(float) / float(fs)
    intervals = intervals[(intervals >= 0.25) & (intervals <= 12.0)]
    if len(intervals) == 0:
        return output
    q25, q75 = np.percentile(intervals, [25, 75])
    output[f"{prefix}_rate_per_min"] = float(60.0 / np.median(intervals))
    output[f"{prefix}_interval_median_s"] = float(np.median(intervals))
    output[f"{prefix}_interval_iqr_s"] = float(q75 - q25)
    output[f"{prefix}_sdnn_s"] = float(np.std(intervals))
    if len(intervals) >= 2:
        output[f"{prefix}_rmssd_s"] = float(np.sqrt(np.mean(np.diff(intervals) ** 2)))
    return output


def _ecg_features(values: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    # A generic peak detector at 0.30 s can count QRS plus T-wave/EMG
    # excursions as separate beats.  Use a QRS-focused band and a refractory
    # period that still supports tachycardia but rejects same-beat double peaks.
    peaks = _peaks(values, fs, 8.0, 20.0, 0.45, prominence_floor=0.75)
    output = _interval_stats(peaks, fs, prefix)
    output.update({f"{prefix}_quality_{key}": value for key, value in _quality(values).items()})
    return output


def _resp_features(values: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    peaks = _peaks(values, fs, 0.05, 1.2, 1.0)
    output = _interval_stats(peaks, fs, prefix)
    output.update({f"{prefix}_quality_{key}": value for key, value in _quality(values).items()})
    return output


def _pulse_features(values: np.ndarray, fs: float, prefix: str, arterial: bool) -> dict[str, float]:
    filled, _ = _filled(values)
    peaks = _peaks(filled, fs, 0.5, 8.0, 0.30)
    output = _interval_stats(peaks, fs, prefix)
    output.update({f"{prefix}_quality_{key}": value for key, value in _quality(values).items()})
    if len(peaks) < 2:
        return output

    systolic = []
    diastolic = []
    means = []
    amplitudes = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        segment = filled[left:right]
        if len(segment) < 2:
            continue
        high = float(np.max(segment))
        low = float(np.min(segment))
        systolic.append(high)
        diastolic.append(low)
        means.append(float(np.mean(segment)))
        amplitudes.append(high - low)
    if amplitudes:
        output[f"{prefix}_pulse_amplitude_median"] = float(np.median(amplitudes))
        output[f"{prefix}_pulse_amplitude_iqr"] = float(np.subtract(*np.percentile(amplitudes, [75, 25])))
    if arterial and systolic:
        output[f"{prefix}_sbp_median"] = float(np.median(systolic))
        output[f"{prefix}_dbp_median"] = float(np.median(diastolic))
        output[f"{prefix}_map_median"] = float(np.median(means))
        output[f"{prefix}_pulse_pressure_median"] = float(np.median(np.asarray(systolic) - np.asarray(diastolic)))
        output[f"{prefix}_hypotension_fraction_lt65"] = float(np.mean(np.asarray(means) < 65.0))
    return output


def beat_level_array_features(
    signal_values: np.ndarray,
    signal_names: list[str] | tuple[str, ...],
    sampling_frequency_hz: float | None = None,
) -> dict[str, float]:
    """Encode each available physiologic channel at the beat/cycle level."""

    if signal_values.ndim != 2:
        raise ValueError("signal_values must be a 2D array")
    if signal_values.shape[1] != len(signal_names):
        raise ValueError("signal name count must match signal array columns")
    fs = float(sampling_frequency_hz or 0.0)
    features: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    for index, signal_name in enumerate(signal_names):
        group = classify_signal_name(signal_name)
        if group not in {"ecg", "arterial_pressure", "pleth", "respiration"} or fs <= 0:
            continue
        group_counts[group] = group_counts.get(group, 0) + 1
        prefix = f"wavebeat_{group}_{group_counts[group]}"
        values = signal_values[:, index]
        if group == "ecg":
            features.update(_ecg_features(values, fs, prefix))
        elif group == "respiration":
            features.update(_resp_features(values, fs, prefix))
        else:
            features.update(_pulse_features(values, fs, prefix, arterial=group == "arterial_pressure"))

    # Collapse redundant leads into an auditable consensus instead of asking
    # the downstream regressor to arbitrate between independently noisy leads.
    for group in group_counts:
        rate_values = [
            value
            for key, value in features.items()
            if key.startswith(f"wavebeat_{group}_")
            and key.endswith("_rate_per_min")
            and np.isfinite(value)
        ]
        if not rate_values:
            continue
        rates = np.asarray(rate_values, dtype=float)
        median = float(np.median(rates))
        features[f"wavebeat_{group}_consensus_rate_per_min"] = median
        features[f"wavebeat_{group}_consensus_lead_count"] = float(len(rates))
        features[f"wavebeat_{group}_consensus_iqr"] = float(np.subtract(*np.percentile(rates, [75, 25])))
        features[f"wavebeat_{group}_consensus_agreement_fraction"] = float(
            np.mean(np.abs(rates - median) <= max(8.0, median * 0.10))
        )
    return features
