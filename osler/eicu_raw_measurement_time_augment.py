"""Attach target-specific next raw-measurement-time labels to eICU cohorts.

The existing eICU transition tables are regularized prediction grids.  They
retain the current value, age, and future value, but not the timestamp of the
next raw event.  This script joins the actual eICU lab/vital/urine event table
using the original ICU-relative offsets and writes a separate augmented copy.

The label is strictly the first event with ``event_offset > anchor_offset``.
It is never inferred from the transition grid and is never used as a current
value.  Outputs are local research artifacts; no row-level data is committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_sepsis_transition_extract import horizon_suffix, read_measurements
from eicu_respiratory_transition_extract import read_respiratory_support_measurements


DEFAULT_MODULES = (
    "dka", "sepsis", "aki", "respiratory",
    "integumentary_skin_wound", "toxic_metabolic", "electrolyte_acid_base",
    "endocrine_stress", "gi_pancreatic_nutrition", "cardiac_injury",
    "musculoskeletal_rhabdo", "immune_inflammatory",
    "cardiovascular_instability", "acute_neuro", "hepatic_failure",
    "coagulopathy_heme",
)
HORIZONS = (1, 3, 6, 12, 24, 48)

# A future label is not equally reliable for every variable.  Fast vitals are
# expected to be measured close to the nominal target time; sparse labs need a
# wider search window, but the actual event delta is retained and exposed to
# the model.  The wider ``near`` band is audit-only by default.
DENSE_VARIABLES = {
    "heart_rate", "map", "o2sat", "respiratory_rate", "temperature",
    "systolic_bp", "diastolic_bp", "glucose", "lactate", "spo2",
}
# Only genuinely high-frequency continuous measurements may receive an
# interpolated *training-label candidate*. Chemistry and sparse laboratories
# deliberately stay raw-event-only: interpolation would turn an unmeasured
# clinical change into an invented smooth trajectory.
INTERPOLATABLE_CONTINUOUS_VARIABLES = {
    "heart_rate", "map", "o2sat", "respiratory_rate", "temperature",
    "systolic_bp", "diastolic_bp", "spo2",
}
VERY_SPARSE_VARIABLES = {
    "troponin", "troponin_i", "troponin_t", "inr", "ptt", "fibrinogen",
    "bilirubin", "bilirubin_direct", "albumin", "ammonia", "tsh", "t4",
    "t3", "cortisol", "acth", "insulin", "hba1c",
}


def _future_tolerances(variable: str, horizon: float) -> tuple[float, float]:
    """Return (strict, audit-near) tolerances in hours for one target."""

    horizon = float(horizon)
    strict = min(2.0, max(0.25, horizon / 2.0))
    if variable in VERY_SPARSE_VARIABLES:
        near = min(24.0, max(strict, horizon * 0.75))
    elif variable in DENSE_VARIABLES:
        near = min(6.0, max(strict, horizon / 2.0))
    else:
        near = min(12.0, max(strict, horizon * 0.60))
    return float(strict), float(near)


def _interpolation_max_gap_hours(variable: str, horizon: float) -> float:
    """Return a conservative bracket width for a continuous vital label.

    This is intentionally a data-quality rule rather than a physiological
    model. It merely permits a label at the nominal time when two close raw
    events bracket that time. A wider interval could hide a true acute change.
    """

    if variable not in INTERPOLATABLE_CONTINUOUS_VARIABLES:
        return 0.0
    return float(min(1.0, max(0.25, float(horizon) / 4.0)))


def _nearest_future_event(
    times: np.ndarray,
    values: np.ndarray,
    anchor: float,
    horizon: float,
    variable: str,
) -> tuple[float, float, str, float, float]:
    """Find a post-anchor raw event and classify its temporal quality."""

    strict, near = _future_tolerances(variable, horizon)
    nominal = float(anchor + horizon)
    first = int(np.searchsorted(times, anchor + 1e-6, side="left"))
    if first >= len(times):
        return np.nan, np.nan, "missing", strict, near
    candidates = np.arange(first, len(times), dtype=np.int64)
    best = int(candidates[np.argmin(np.abs(times[candidates] - nominal))])
    error = abs(float(times[best]) - nominal)
    if error <= strict + 1e-6:
        quality = "exact"
    elif error <= near + 1e-6:
        quality = "near"
    else:
        return np.nan, np.nan, "missing", strict, near
    return (
        float(values[best]),
        float(times[best]),
        quality,
        strict,
        near,
    )


def _future_label_arrays(
    anchors: pd.DataFrame,
    variable: str,
    event_lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    future_label_mode: str = "raw_exact",
    interpolated_label_weight: float = 0.50,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Vectorized raw-event labels for unique stay/anchor/horizon rows."""

    count = len(anchors)
    values = np.full(count, np.nan, dtype=np.float64)
    event_times = np.full(count, np.nan, dtype=np.float64)
    quality = np.full(count, "missing", dtype=object)
    strict = np.full(count, np.nan, dtype=np.float64)
    near = np.full(count, np.nan, dtype=np.float64)
    deltas = np.full(count, np.nan, dtype=np.float64)
    left_event_times = np.full(count, np.nan, dtype=np.float64)
    right_event_times = np.full(count, np.nan, dtype=np.float64)
    interpolation_gaps = np.full(count, np.nan, dtype=np.float64)
    weights = np.zeros(count, dtype=np.float64)
    anchor_values = _anchor_hours(anchors)
    stays = pd.to_numeric(anchors["stay_id"], errors="coerce").to_numpy(dtype=np.float64)
    horizons = pd.to_numeric(anchors["_augment_horizon"], errors="coerce").to_numpy(dtype=np.float64)
    for (stay, horizon), positions in anchors.groupby(
        ["stay_id", "_augment_horizon"], sort=False
    ).groups.items():
        pair = event_lookup.get(int(stay))
        if pair is None:
            continue
        times, raw_values = pair
        positions = np.asarray(positions, dtype=np.int64)
        anchor = anchor_values[positions]
        nominal = anchor + float(horizon)
        right = np.searchsorted(times, nominal, side="left")
        left = right - 1
        candidates = np.full((len(positions), 2), -1, dtype=np.int64)
        candidates[:, 0] = np.where(right < len(times), right, -1)
        candidates[:, 1] = np.where(left >= 0, left, -1)
        candidate_times = np.full(candidates.shape, np.nan, dtype=np.float64)
        valid_index = candidates >= 0
        candidate_times[valid_index] = times[candidates[valid_index]]
        valid_index &= candidate_times > (anchor[:, None] + 1e-6)
        errors = np.where(valid_index, np.abs(candidate_times - nominal[:, None]), np.inf)
        choice = np.argmin(errors, axis=1)
        best_error = errors[np.arange(len(positions)), choice]
        best_index = candidates[np.arange(len(positions)), choice]
        strict_tol, near_tol = _future_tolerances(variable, float(horizon))
        strict[positions] = strict_tol
        near[positions] = near_tol
        accepted = (best_index >= 0) & np.isfinite(best_error) & (best_error <= near_tol + 1e-6)
        if not accepted.any():
            continue
        selected = positions[accepted]
        selected_index = best_index[accepted]
        selected_error = best_error[accepted]
        values[selected] = raw_values[selected_index]
        event_times[selected] = times[selected_index]
        deltas[selected] = event_times[selected] - anchor_values[selected]
        quality[selected] = np.where(selected_error <= strict_tol + 1e-6, "exact", "near")
        weights[selected] = np.where(
            selected_error <= strict_tol + 1e-6, 1.0, 0.0
        )

        if future_label_mode != "interpolate_dense":
            continue
        max_gap = _interpolation_max_gap_hours(variable, horizon)
        if max_gap <= 0.0:
            continue
        left_times = np.full(len(positions), np.nan, dtype=np.float64)
        right_times = np.full(len(positions), np.nan, dtype=np.float64)
        left_values = np.full(len(positions), np.nan, dtype=np.float64)
        right_values = np.full(len(positions), np.nan, dtype=np.float64)
        valid_left = left >= 0
        valid_right = right < len(times)
        left_times[valid_left] = times[left[valid_left]]
        right_times[valid_right] = times[right[valid_right]]
        left_values[valid_left] = raw_values[left[valid_left]]
        right_values[valid_right] = raw_values[right[valid_right]]
        gaps = right_times - left_times
        can_interpolate = (
            np.isfinite(left_times)
            & np.isfinite(right_times)
            & (left_times > anchor + 1e-6)
            & (right_times > left_times)
            & (gaps <= max_gap + 1e-6)
            # Keep a raw event that is already exactly at the nominal target.
            # Otherwise a tight bracket removes residual timing jitter from an
            # "exact"-tolerance label without widening the observation gap.
            & (np.abs(deltas[positions] - float(horizon)) > 1e-6)
        )
        if can_interpolate.any():
            chosen = positions[can_interpolate]
            ratio = (nominal[can_interpolate] - left_times[can_interpolate]) / gaps[can_interpolate]
            values[chosen] = left_values[can_interpolate] + ratio * (
                right_values[can_interpolate] - left_values[can_interpolate]
            )
            # The target is reconstructed at the nominal time, not falsely
            # presented as a raw event timestamp.
            event_times[chosen] = nominal[can_interpolate]
            deltas[chosen] = float(horizon)
            quality[chosen] = "interpolated_dense"
            weights[chosen] = float(interpolated_label_weight)
            left_event_times[chosen] = left_times[can_interpolate]
            right_event_times[chosen] = right_times[can_interpolate]
            interpolation_gaps[chosen] = gaps[can_interpolate]
    return (
        values, event_times, deltas, quality, strict, near, weights,
        left_event_times, right_event_times, interpolation_gaps,
    )


def _transition_paths(root: Path, modules: tuple[str, ...], horizons: tuple[int, ...]):
    for module in modules:
        for horizon in horizons:
            path = root / f"eicu_{module}_transitions_{horizon}h.parquet"
            if path.exists():
                yield module, horizon, path


def _anchor_hours(frame: pd.DataFrame) -> np.ndarray:
    if "_anchor_hour" in frame.columns:
        return pd.to_numeric(frame["_anchor_hour"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    onset = pd.to_numeric(frame.get("onset_hour", 0.0), errors="coerce").fillna(0.0)
    elapsed = pd.to_numeric(frame.get("hours_since_onset", 0.0), errors="coerce").fillna(0.0)
    return (onset + elapsed).to_numpy(dtype=np.float64)


def _next_times_from_lookup(anchors: pd.DataFrame, event_times: dict[int, np.ndarray]) -> np.ndarray:
    result = np.full(len(anchors), np.nan, dtype=np.float64)
    anchor_hours = _anchor_hours(anchors)
    for stay, positions in anchors.groupby("stay_id", sort=False).groups.items():
        times = event_times.get(int(stay))
        if times is None or len(times) == 0:
            continue
        positions = np.asarray(positions, dtype=np.int64)
        indices = np.searchsorted(times, anchor_hours[positions], side="right")
        valid = indices < len(times)
        if valid.any():
            selected = positions[valid]
            result[selected] = times[indices[valid]] - anchor_hours[selected]
    return result


def augment_frame(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    horizon: float,
    future_label_mode: str = "raw_exact",
    interpolated_label_weight: float = 0.50,
) -> pd.DataFrame:
    frame = frame.copy()
    variables = sorted({
        column[:-2] for column in frame.columns
        if column.endswith("_t") and f"{column[:-2]}_age_hr" in frame.columns
    })
    frame["next_measurement_time_label_source"] = "raw_eicu_event_offset_strictly_after_anchor"
    frame["future_label_source"] = "raw_eicu_event_value_nearest_nominal_target"
    for variable in variables:
        scoped = events[events["var"] == variable]
        # Build both arrays from the same cleaned rows. If a raw event has a
        # timestamp but no numeric value, dropping it from only one array
        # would shift every later value onto the wrong timestamp.
        event_times = {}
        event_values = {}
        for stay, group in scoped.groupby("stay_id", sort=False):
            ordered = (
                group.assign(
                    time_hr=pd.to_numeric(group["time_hr"], errors="coerce"),
                    valuenum=pd.to_numeric(group["valuenum"], errors="coerce"),
                )
                .dropna(subset=["time_hr", "valuenum"])
                .sort_values("time_hr")
            )
            if ordered.empty:
                continue
            event_times[int(stay)] = ordered["time_hr"].to_numpy(dtype=np.float64)
            event_values[int(stay)] = ordered["valuenum"].to_numpy(dtype=np.float64)
        values = _next_times_from_lookup(frame, event_times)
        frame[f"next_measurement_time_hr_{variable}"] = values
        frame[f"next_measurement_observed_{variable}"] = np.isfinite(values)
        lookup = {stay: (event_times[stay], event_values[stay]) for stay in event_times}
        anchors = frame[["stay_id"]].copy()
        anchors["_anchor_hour"] = _anchor_hours(frame)
        anchors["_augment_horizon"] = float(horizon)
        (
            future_values,
            future_times,
            future_deltas,
            quality,
            strict,
            near,
            weights,
            left_times,
            right_times,
            interpolation_gaps,
        ) = _future_label_arrays(
            anchors,
            variable,
            lookup,
            future_label_mode=future_label_mode,
            interpolated_label_weight=interpolated_label_weight,
        )
        suffix = horizon_suffix(horizon)
        future_column = f"{variable}_{suffix}"
        if future_column in frame.columns:
            # Rebuild the future value from raw events.  This prevents a
            # regularized transition grid from becoming the hidden label.
            frame[future_column] = future_values
        frame[f"future_event_time_hr_{variable}"] = future_times
        frame[f"future_delta_t_hr_{variable}"] = future_deltas
        frame[f"future_label_quality_{variable}"] = quality
        frame[f"future_label_weight_{variable}"] = weights
        frame[f"future_label_tolerance_strict_hr_{variable}"] = strict
        frame[f"future_label_tolerance_near_hr_{variable}"] = near
        frame[f"future_label_left_event_time_hr_{variable}"] = left_times
        frame[f"future_label_right_event_time_hr_{variable}"] = right_times
        frame[f"future_label_interpolation_gap_hr_{variable}"] = interpolation_gaps
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--transition-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modules", default=",".join(DEFAULT_MODULES))
    parser.add_argument("--horizons", default=",".join(str(v) for v in HORIZONS))
    parser.add_argument(
        "--future-label-mode",
        choices=("raw_exact", "interpolate_dense"),
        default="raw_exact",
        help="Use raw-only labels or a conservative bracketed-label candidate for continuous vitals.",
    )
    parser.add_argument(
        "--interpolated-label-weight",
        type=float,
        default=0.50,
        help="Candidate loss weight for an interpolated continuous-vital label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.interpolated_label_weight) <= 1.0:
        raise ValueError("interpolated-label-weight must be in (0, 1]")
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    paths = list(_transition_paths(args.transition_root, modules, horizons))
    if not paths:
        raise FileNotFoundError("No eICU transition parquet files found")

    frames = []
    stay_ids: set[int] = set()
    for module, horizon, path in paths:
        frame = pd.read_parquet(path)
        frame["_augment_module"] = module
        frame["_augment_horizon"] = horizon
        frames.append((module, horizon, path, frame))
        stay_ids.update(pd.to_numeric(frame["stay_id"], errors="coerce").dropna().astype(int).unique())
    print(f"Transition files: {len(frames)}; unique stays: {len(stay_ids):,}", flush=True)
    events = read_measurements(stay_ids)
    respiratory_events = read_respiratory_support_measurements(args.data_root, stay_ids)
    if len(respiratory_events):
        respiratory_events = respiratory_events.copy()
        respiratory_events["time_hr"] = (
            pd.to_numeric(respiratory_events["offset"], errors="coerce") / 60.0
        )
        respiratory_events["valuenum"] = pd.to_numeric(
            respiratory_events["valuenum"], errors="coerce"
        )
        respiratory_events = respiratory_events[
            ["stay_id", "var", "time_hr", "valuenum"]
        ].dropna(subset=["time_hr", "valuenum"])
        events = pd.concat([events, respiratory_events], ignore_index=True)
    if len(events):
        events = events[["stay_id", "var", "time_hr", "valuenum"]].copy()
        events["var"] = events["var"].astype(str)
    print(f"Raw measurement events: {len(events):,}; variables: {events['var'].nunique() if len(events) else 0}", flush=True)

    # Compute labels once per unique stay/anchor, then reuse them for every
    # horizon file.  The former implementation rebuilt the same event lookup
    # for every horizon and was needlessly quadratic in the number of files.
    anchor_parts = []
    future_anchor_parts = []
    all_variables = set()
    for _, horizon, _, frame in frames:
        frame = frame.copy()
        frame["_anchor_hour"] = _anchor_hours(frame)
        anchor_parts.append(frame[["stay_id", "_anchor_hour"]])
        future_anchor = frame[["stay_id", "_anchor_hour"]].copy()
        future_anchor["_augment_horizon"] = float(horizon)
        future_anchor_parts.append(future_anchor)
        all_variables.update({
            column[:-2] for column in frame.columns
            if column.endswith("_t") and f"{column[:-2]}_age_hr" in frame.columns
        })
    unique_anchors = pd.concat(anchor_parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
    future_anchors = (
        pd.concat(future_anchor_parts, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    time_label_table = unique_anchors.copy()
    time_label_table["next_measurement_time_label_source"] = "raw_eicu_event_offset_strictly_after_anchor"
    future_label_table = future_anchors.copy()
    future_label_table["future_label_source"] = (
        "raw_eicu_event_value_nearest_nominal_target"
        if args.future_label_mode == "raw_exact"
        else "raw_eicu_event_or_dense_bracket_interpolation"
    )
    event_lookup_by_variable = {}
    for variable in sorted(all_variables):
        scoped = events[events["var"] == variable]
        event_times = {
            int(stay): np.sort(pd.to_numeric(group["time_hr"], errors="coerce").dropna().to_numpy(dtype=np.float64))
            for stay, group in scoped.groupby("stay_id", sort=False)
        }
        time_label_table[f"next_measurement_time_hr_{variable}"] = _next_times_from_lookup(
            unique_anchors, event_times
        )
        time_label_table[f"next_measurement_observed_{variable}"] = np.isfinite(
            time_label_table[f"next_measurement_time_hr_{variable}"]
        )
        event_lookup_by_variable[variable] = {
            int(stay): (
                ordered["time_hr"].to_numpy(dtype=np.float64),
                ordered["valuenum"].to_numpy(dtype=np.float64),
            )
            for stay, group in scoped.groupby("stay_id", sort=False)
            for ordered in [
                group.assign(
                    time_hr=pd.to_numeric(group["time_hr"], errors="coerce"),
                    valuenum=pd.to_numeric(group["valuenum"], errors="coerce"),
                )
                .dropna(subset=["time_hr", "valuenum"])
                .sort_values("time_hr")
            ]
        }
        (
            values,
            event_time,
            deltas,
            quality,
            strict,
            near,
            weights,
            left_times,
            right_times,
            interpolation_gaps,
        ) = _future_label_arrays(
            future_anchors,
            variable,
            event_lookup_by_variable[variable],
            future_label_mode=args.future_label_mode,
            interpolated_label_weight=float(args.interpolated_label_weight),
        )
        future_label_table[f"future_raw_value_{variable}"] = values
        future_label_table[f"future_event_time_hr_{variable}"] = event_time
        future_label_table[f"future_delta_t_hr_{variable}"] = deltas
        future_label_table[f"future_label_quality_{variable}"] = quality
        future_label_table[f"future_label_weight_{variable}"] = weights
        future_label_table[f"future_label_tolerance_strict_hr_{variable}"] = strict
        future_label_table[f"future_label_tolerance_near_hr_{variable}"] = near
        future_label_table[f"future_label_left_event_time_hr_{variable}"] = left_times
        future_label_table[f"future_label_right_event_time_hr_{variable}"] = right_times
        future_label_table[f"future_label_interpolation_gap_hr_{variable}"] = interpolation_gaps
        print(json.dumps({
            "label_variable": variable,
            "strict_observed": int((quality == "exact").sum()),
            "interpolated_observed": int((quality == "interpolated_dense").sum()),
            "near_observed": int((quality == "near").sum()),
        }), flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for module, horizon, path, frame in frames:
        frame = frame.copy()
        frame["_anchor_hour"] = _anchor_hours(frame)
        variables = sorted({
            column[:-2] for column in frame.columns
            if column.endswith("_t") and f"{column[:-2]}_age_hr" in frame.columns
        })
        label_columns = ["stay_id", "_anchor_hour", "next_measurement_time_label_source"]
        label_columns += [f"next_measurement_time_hr_{v}" for v in variables]
        label_columns += [f"next_measurement_observed_{v}" for v in variables]
        augmented = frame.merge(
            time_label_table[label_columns], on=["stay_id", "_anchor_hour"], how="left", sort=False,
        )
        future_columns = ["stay_id", "_anchor_hour", "_augment_horizon", "future_label_source"]
        for variable in variables:
            future_columns += [
                f"future_raw_value_{variable}",
                f"future_event_time_hr_{variable}",
                f"future_delta_t_hr_{variable}",
                f"future_label_quality_{variable}",
                f"future_label_weight_{variable}",
                f"future_label_tolerance_strict_hr_{variable}",
                f"future_label_tolerance_near_hr_{variable}",
                f"future_label_left_event_time_hr_{variable}",
                f"future_label_right_event_time_hr_{variable}",
                f"future_label_interpolation_gap_hr_{variable}",
            ]
        augmented = augmented.merge(
            future_label_table[future_columns],
            on=["stay_id", "_anchor_hour", "_augment_horizon"],
            how="left",
            sort=False,
        )
        suffix = horizon_suffix(horizon)
        for variable in variables:
            future_column = f"{variable}_{suffix}"
            raw_column = f"future_raw_value_{variable}"
            if future_column in augmented.columns and raw_column in augmented.columns:
                augmented[future_column] = pd.to_numeric(
                    augmented[raw_column], errors="coerce"
                )
        augmented.drop(
            columns=["_augment_module", "_augment_horizon", "_anchor_hour"]
            + [column for column in augmented.columns if column.startswith("future_raw_value_")],
            inplace=True,
            errors="ignore",
        )
        output = args.output_root / path.name
        augmented.to_parquet(output, index=False)
        label_columns = [column for column in augmented if column.startswith("next_measurement_time_hr_")]
        reports.append({
            "module": module,
            "horizon_hours": horizon,
            "rows": int(len(augmented)),
            "stays": int(augmented["stay_id"].nunique()),
            "output": str(output.resolve()),
            "label_columns": len(label_columns),
            "observed_label_counts": {
                column.removeprefix("next_measurement_time_hr_"): int(augmented[column].notna().sum())
                for column in label_columns
            },
        })
        print(json.dumps(reports[-1]), flush=True)
    report = {
        "schema": "eicu_raw_measurement_time_augmentation.v2",
        "data_root": str(args.data_root.resolve()),
        "transition_root": str(args.transition_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "raw_event_source": (
            "eicu/lab.csv.gz + vitalPeriodic.csv.gz + vitalAperiodic.csv.gz + "
            "intakeOutput.csv.gz + respiratoryCharting.csv.gz"
        ),
        "strict_timestamp_rule": "event_offset > anchor_offset",
        "future_label_rule": (
            "nearest raw event strictly after anchor; exact labels are within the "
            "target-specific strict tolerance; near labels are audit-only; dense "
            "continuous variables may use a separately marked bracketed-label candidate"
        ),
        "future_delta_t_columns": "future_delta_t_hr_<variable>",
        "future_quality_columns": "future_label_quality_<variable>",
        "future_label_mode": args.future_label_mode,
        "interpolated_label_weight": float(args.interpolated_label_weight),
        "training_label_weight_rule": (
            "exact=1.0; interpolated_dense=candidate_weight; near/missing=0.0"
            if args.future_label_mode == "interpolate_dense"
            else "exact=1.0; near/missing=0.0"
        ),
        "row_level_outputs_committed": False,
        "causal_claim_allowed": False,
        "files": reports,
    }
    report_path = args.output_root / "eicu_raw_measurement_time_augmentation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path.resolve()), "files": len(reports)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
