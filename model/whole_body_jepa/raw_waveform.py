"""Fixed-width, measurement-pure raw waveform tensors for JEPA training."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .waveform_event import signal_group


SIGNAL_GROUPS = ("ecg", "pleth", "respiration", "arterial_pressure")
PAST_SCALES_SECONDS = (60, 300, 900)


def _flatline_fraction(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return 1.0
    scale = max(float(np.std(finite)), 1e-8)
    return float(np.mean(np.abs(np.diff(finite)) <= scale * 1e-4))


def _resample(values: np.ndarray, length: int) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    valid_fraction = float(finite.mean()) if len(values) else 0.0
    if int(finite.sum()) < 2:
        return np.zeros(length, dtype=np.float32), valid_fraction
    source = np.linspace(0.0, 1.0, len(values), dtype=float)
    target = np.linspace(0.0, 1.0, length, dtype=float)
    interpolated = np.interp(target, source[finite], values[finite])
    return interpolated.astype(np.float32), valid_fraction


def _channel_priority(name: str, group: str) -> int:
    value = str(name).strip().upper()
    if group == "ecg" and value in {"II", "ECG II", "ECG2"}:
        return 3
    if group == "arterial_pressure" and value in {"ABP", "ART", "AO"}:
        return 3
    if group == "pleth" and "PLETH" in value:
        return 2
    if group == "respiration" and value in {"RESP", "RESP IMP"}:
        return 2
    return 1


def _representative_channels(
    values: np.ndarray, names: list[str] | tuple[str, ...]
) -> dict[str, int]:
    output = {}
    for group in SIGNAL_GROUPS:
        candidates = []
        for index, name in enumerate(names):
            if signal_group(name) != group:
                continue
            valid = float(np.isfinite(values[:, index]).mean())
            candidates.append((_channel_priority(name, group), valid, -index, index))
        if candidates:
            output[group] = max(candidates)[-1]
    return output


def build_raw_waveform_tensor(
    values: np.ndarray,
    names: list[str] | tuple[str, ...],
    sampling_frequency_hz: float,
    anchor_sample: int,
    *,
    past_scales_seconds: tuple[int, ...] = PAST_SCALES_SECONDS,
    points_per_scale: int = 256,
    future_seconds: int = 60,
) -> dict[str, np.ndarray]:
    """Build the canonical tensor from already aligned waveform arrays."""

    values = np.asarray(values, dtype=float)
    fs = float(sampling_frequency_hz)
    if not past_scales_seconds or any(seconds <= 0 for seconds in past_scales_seconds):
        raise ValueError("past waveform scales must contain positive seconds")
    past_count = int(round(max(past_scales_seconds) * fs))
    future_count = int(round(future_seconds * fs))
    start = int(anchor_sample) - past_count
    stop = int(anchor_sample) + future_count
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("waveform values and names do not have matching channels")
    if start < 0 or stop > len(values):
        raise ValueError("waveform array does not contain full past/future window")
    past_values = values[start:int(anchor_sample)]
    future_values = values[int(anchor_sample):stop]
    selected = _representative_channels(past_values, names)

    past = np.zeros(
        (len(past_scales_seconds), len(SIGNAL_GROUPS), points_per_scale),
        dtype=np.float32,
    )
    future = np.zeros((len(SIGNAL_GROUPS), points_per_scale), dtype=np.float32)
    past_mask = np.zeros((len(past_scales_seconds), len(SIGNAL_GROUPS)), dtype=np.float32)
    future_mask = np.zeros(len(SIGNAL_GROUPS), dtype=np.float32)
    quality = np.zeros((len(past_scales_seconds), len(SIGNAL_GROUPS), 2), dtype=np.float32)

    for group_index, group in enumerate(SIGNAL_GROUPS):
        channel = selected.get(group)
        if channel is None:
            continue
        reference = past_values[:, channel]
        finite = reference[np.isfinite(reference)]
        if len(finite) < 32:
            continue
        median = float(np.median(finite))
        q25, q75 = np.percentile(finite, (25, 75))
        scale = max(float(q75 - q25), float(np.std(finite)) * 0.5, 1e-5)
        normalized_past = np.clip((reference - median) / scale, -8.0, 8.0)
        normalized_future = np.clip(
            (future_values[:, channel] - median) / scale, -8.0, 8.0
        )
        for scale_index, seconds in enumerate(past_scales_seconds):
            count = int(round(seconds * fs))
            sampled, valid_fraction = _resample(
                normalized_past[-count:], points_per_scale
            )
            past[scale_index, group_index] = sampled
            past_mask[scale_index, group_index] = float(valid_fraction >= 0.80)
            quality[scale_index, group_index] = (
                valid_fraction,
                _flatline_fraction(reference[-count:]),
            )
        sampled_future, valid_fraction = _resample(
            normalized_future, points_per_scale
        )
        future[group_index] = sampled_future
        future_mask[group_index] = float(valid_fraction >= 0.80)

    if past_mask[:, :3].max() <= 0:
        raise ValueError("no ECG, pleth, or respiration context channel")
    if future_mask[:3].max() <= 0:
        raise ValueError("no future ECG, pleth, or respiration target channel")
    return {
        "past": past,
        "future": future,
        "past_mask": past_mask,
        "future_mask": future_mask,
        "quality": quality,
    }


def read_raw_waveform_tensor(
    record_path: str | Path,
    anchor_sample: int,
    *,
    past_scales_seconds: tuple[int, ...] = PAST_SCALES_SECONDS,
    points_per_scale: int = 256,
    future_seconds: int = 60,
) -> dict[str, np.ndarray]:
    """Read past multi-scale context and a post-anchor future waveform target."""

    import wfdb

    path = Path(record_path)
    header = wfdb.rdheader(str(path))
    fs = float(header.fs)
    past_count = int(round(max(past_scales_seconds) * fs))
    future_count = int(round(future_seconds * fs))
    start = int(anchor_sample) - past_count
    stop = int(anchor_sample) + future_count
    if start < 0 or header.sig_len is None or stop > int(header.sig_len):
        raise ValueError("waveform segment does not contain full past/future window")
    record = wfdb.rdrecord(str(path), sampfrom=start, sampto=stop, physical=True)
    return build_raw_waveform_tensor(
        np.asarray(record.p_signal, dtype=float),
        record.sig_name,
        fs,
        past_count,
        past_scales_seconds=past_scales_seconds,
        points_per_scale=points_per_scale,
        future_seconds=future_seconds,
    )
