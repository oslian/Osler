"""Portable pre-anchor waveform features and incident hypoxemia labels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


EVENT_THRESHOLD = 92.0
WAVEFORM_SCALES_SECONDS = (60, 300, 900)


@dataclass(frozen=True)
class SustainedHypoxemiaDefinition:
    """Cross-database event definition that rejects isolated monitor artifacts."""

    threshold: float = EVENT_THRESHOLD
    horizon_seconds: float = 3600.0
    stable_lookback_seconds: float = 600.0
    max_low_pair_gap_seconds: float = 300.0
    min_low_pair_separation_seconds: float = 30.0
    minimum_future_observations: int = 2


def incident_sustained_hypoxemia(
    times_seconds: np.ndarray,
    spo2: np.ndarray,
    anchor_seconds: float,
    definition: SustainedHypoxemiaDefinition = SustainedHypoxemiaDefinition(),
) -> tuple[int | None, dict[str, float]]:
    """Label a new sustained low-SpO2 event after an anchor.

    The anchor is eligible only when recent observed SpO2 is above threshold.
    A future event requires two low readings separated by 30 seconds to five
    minutes, which is portable to minute-level ICU tables and rejects one-sample
    waveform monitor artifacts.
    """

    times = np.asarray(times_seconds, dtype=float)
    values = np.asarray(spo2, dtype=float)
    valid = np.isfinite(times) & np.isfinite(values) & (values >= 30.0) & (values <= 100.0)
    times = times[valid]
    values = values[valid]
    if not len(times):
        return None, {}
    if np.any(np.diff(times) < 0):
        order = np.argsort(times, kind="stable")
        times = times[order]
        values = values[order]
    past_left = int(
        np.searchsorted(
            times, anchor_seconds - definition.stable_lookback_seconds, side="left"
        )
    )
    past_right = int(np.searchsorted(times, anchor_seconds, side="right"))
    future_left = past_right
    future_right = int(
        np.searchsorted(times, anchor_seconds + definition.horizon_seconds, side="right")
    )
    if past_right <= past_left or future_right - future_left < definition.minimum_future_observations:
        return None, {}
    past_values = values[past_left:past_right]
    if float(past_values[-1]) <= definition.threshold or np.any(
        past_values <= definition.threshold
    ):
        return None, {}

    future_times = times[future_left:future_right]
    future_values = values[future_left:future_right]
    low_times = future_times[future_values <= definition.threshold]
    event = False
    if len(low_times) >= 2:
        for left, start_time in enumerate(low_times[:-1]):
            right = int(
                np.searchsorted(
                    low_times,
                    start_time + definition.min_low_pair_separation_seconds,
                    side="left",
                )
            )
            if right < len(low_times) and (
                low_times[right] - start_time
                <= definition.max_low_pair_gap_seconds
            ):
                event = True
                break
    descriptor = {
        "future_spo2_min": float(np.min(future_values)),
        "future_spo2_mean": float(np.mean(future_values)),
        "future_spo2_std": float(np.std(future_values)),
        "future_low_fraction": float(np.mean(future_values <= definition.threshold)),
        "future_observation_count": float(len(future_values)),
    }
    return int(event), descriptor


def signal_group(name: str) -> str | None:
    value = str(name).strip().upper()
    if value.startswith("ECG") or value in {"I", "II", "III", "V", "MCL1"}:
        return "ecg"
    if "PLETH" in value:
        return "pleth"
    if value in {"RESP", "RESP IMP", "RESPIRATION"}:
        return "respiration"
    if value.startswith("ABP") or value in {"ART", "AO"}:
        return "arterial_pressure"
    return None


def _quality(values: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0, 1.0
    clean = values[finite]
    if len(clean) < 2:
        return float(finite.mean()), 1.0
    scale = max(float(np.std(clean)), 1e-9)
    flatline = float(np.mean(np.abs(np.diff(clean)) <= scale * 1e-4))
    return float(finite.mean()), flatline


def _bandpass(values: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values), dtype=float)
    filled = np.where(finite, values, float(np.nanmedian(values[finite])))
    nyquist = fs / 2.0
    if len(filled) < 32 or fs <= 2.0 or high >= nyquist:
        return filled - float(np.median(filled))
    sos = butter(3, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, filled)
    except ValueError:
        return filled - float(np.median(filled))


def _rate_features(values: np.ndarray, fs: float, group: str) -> dict[str, float]:
    if group == "ecg":
        filtered = _bandpass(values, fs, 8.0, 20.0)
        minimum_interval = 0.45
        prominence = 0.75
    elif group == "respiration":
        filtered = _bandpass(values, fs, 0.05, 1.2)
        minimum_interval = 1.0
        prominence = 0.35
    else:
        filtered = _bandpass(values, fs, 0.5, min(8.0, fs * 0.40))
        minimum_interval = 0.30
        prominence = 0.35
    scale = float(np.std(filtered))
    if scale <= 1e-9:
        return {"rate": math.nan, "rate_iqr": math.nan, "rmssd": math.nan}
    normalized = filtered / scale
    peaks, _ = find_peaks(
        normalized,
        distance=max(1, int(round(minimum_interval * fs))),
        prominence=prominence,
    )
    intervals = np.diff(peaks).astype(float) / fs
    intervals = intervals[(intervals >= 0.25) & (intervals <= 12.0)]
    if not len(intervals):
        return {"rate": math.nan, "rate_iqr": math.nan, "rmssd": math.nan}
    rates = 60.0 / intervals
    return {
        "rate": float(np.median(rates)),
        "rate_iqr": float(np.subtract(*np.percentile(rates, [75, 25]))),
        "rmssd": float(np.sqrt(np.mean(np.diff(intervals) ** 2))) if len(intervals) > 1 else math.nan,
    }


def fixed_width_waveform_features(
    signal_values: np.ndarray,
    signal_names: Iterable[str],
    sampling_frequency_hz: float,
    *,
    prefix: str,
) -> dict[str, float]:
    """Create fixed-width group consensus features for one waveform scale."""

    names = tuple(str(value) for value in signal_names)
    output: dict[str, float] = {}
    for group in ("ecg", "pleth", "respiration", "arterial_pressure"):
        indices = [index for index, name in enumerate(names) if signal_group(name) == group]
        rates = []
        valid = []
        flatline = []
        medians = []
        variability = []
        hypotension = []
        for index in indices:
            values = np.asarray(signal_values[:, index], dtype=float)
            finite = values[np.isfinite(values)]
            quality, flat = _quality(values)
            valid.append(quality)
            flatline.append(flat)
            if len(finite):
                medians.append(float(np.median(finite)))
                variability.append(float(np.subtract(*np.percentile(finite, [75, 25]))))
                if group == "arterial_pressure":
                    hypotension.append(float(np.mean(finite < 65.0)))
            rate = _rate_features(values, sampling_frequency_hz, group)["rate"]
            if np.isfinite(rate):
                rates.append(float(rate))
        stem = f"{prefix}_{group}"
        output[f"{stem}_available"] = float(bool(indices))
        output[f"{stem}_lead_count"] = float(len(indices))
        output[f"{stem}_valid_fraction"] = float(np.median(valid)) if valid else 0.0
        output[f"{stem}_flatline_fraction"] = float(np.median(flatline)) if flatline else 1.0
        output[f"{stem}_rate"] = float(np.median(rates)) if rates else math.nan
        output[f"{stem}_rate_agreement"] = (
            float(np.mean(np.abs(np.asarray(rates) - np.median(rates)) <= 10.0))
            if rates
            else math.nan
        )
        output[f"{stem}_level"] = float(np.median(medians)) if medians else math.nan
        output[f"{stem}_variability"] = float(np.median(variability)) if variability else math.nan
        output[f"{stem}_hypotension_fraction"] = (
            float(np.median(hypotension)) if hypotension else math.nan
        )
    return output


def read_multiscale_waveform_features(
    record_path: str | Path,
    anchor_sample: int,
    *,
    scales_seconds: tuple[int, ...] = WAVEFORM_SCALES_SECONDS,
) -> dict[str, float]:
    """Read the longest pre-anchor window once and encode every requested scale."""

    import wfdb

    path = Path(record_path)
    header = wfdb.rdheader(str(path))
    fs = float(header.fs)
    maximum = int(round(max(scales_seconds) * fs))
    start = max(0, int(anchor_sample) - maximum)
    if int(anchor_sample) - start < int(round(min(scales_seconds) * fs)):
        raise ValueError("insufficient pre-anchor waveform history")
    record = wfdb.rdrecord(str(path), sampfrom=start, sampto=int(anchor_sample), physical=True)
    values = np.asarray(record.p_signal, dtype=float)
    output: dict[str, float] = {}
    for seconds in scales_seconds:
        count = min(len(values), int(round(seconds * fs)))
        scale_values = values[-count:]
        output.update(
            fixed_width_waveform_features(
                scale_values,
                record.sig_name,
                fs,
                prefix=f"wf_{seconds}s",
            )
        )
    output["waveform_history_seconds"] = float(len(values) / fs)
    return output


def oxygen_support_level(value: object) -> float:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return math.nan
    if any(token in text for token in ("vent", "ett", "endotracheal", "trach")):
        return 4.0
    if any(token in text for token in ("bipap", "cpap", "non-invasive")):
        return 3.0
    if any(token in text for token in ("high flow", "hfnc", "mask")):
        return 2.0
    if any(token in text for token in ("nasal", "cannula")):
        return 1.0
    if "room air" in text:
        return 0.0
    return math.nan


def respiratory_distress(value: object) -> float:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return math.nan
    abnormal = ("labored", "accessory", "dyspnea", "distress", "agonal", "paradox")
    normal = ("unlabored", "normal", "no distress")
    if any(token in text for token in normal):
        return 0.0
    return float(any(token in text for token in abnormal))


def signed_leading_score(value: object) -> float:
    """Parse the first signed score embedded in a bedside scale label."""

    match = re.search(r"[+-]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else math.nan


def cough_reflex_score(value: object) -> float:
    text = str(value).strip().lower()
    return {"absent": 0.0, "impaired": 1.0, "intact": 2.0}.get(text, math.nan)


def cough_effort_score(value: object) -> float:
    text = str(value).strip().lower()
    return {"none": 0.0, "weak": 1.0, "strong": 2.0}.get(text, math.nan)


def productive_cough(value: object) -> float:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return math.nan
    return float(text.startswith("productive"))


def waveform_event_decision(
    candidate_probability: float,
    *,
    ensemble_standard_deviation: float,
    quality: dict[str, float],
    external_waveform_lockbox_validated: bool,
    calibration_validated: bool,
) -> dict[str, object]:
    """Fail closed unless signal, model, calibration, and external gates pass."""

    reasons = []
    pleth_valid = float(quality.get("wf_300s_pleth_valid_fraction", 0.0))
    pleth_flat = float(quality.get("wf_300s_pleth_flatline_fraction", 1.0))
    ecg_valid = float(quality.get("wf_300s_ecg_valid_fraction", 0.0))
    respiration_valid = float(quality.get("wf_300s_respiration_valid_fraction", 0.0))
    if pleth_valid < 0.95 or pleth_flat > 0.20:
        reasons.append("insufficient_pleth_signal_quality")
    if max(ecg_valid, respiration_valid) < 0.80:
        reasons.append("insufficient_corroborating_waveform")
    if not np.isfinite(ensemble_standard_deviation) or ensemble_standard_deviation > 0.08:
        reasons.append("model_disagreement_too_high")
    if not calibration_validated:
        reasons.append("event_probability_not_calibrated")
    if not external_waveform_lockbox_validated:
        reasons.append("external_waveform_lockbox_not_validated")
    abstained = bool(reasons)
    return {
        "event_risk_probability": None if abstained else float(candidate_probability),
        "candidate_probability": float(candidate_probability),
        "abstained": abstained,
        "abstention_reasons": reasons,
        "causal_claim_allowed": False,
        "clinical_decision_allowed": False,
    }
