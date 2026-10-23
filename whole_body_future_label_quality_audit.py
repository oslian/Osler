"""Audit raw-event future labels before they are used for JEPA training.

The audit is deliberately independent of model code.  It checks that every
target has a real post-anchor event, that the event timing agrees with the
target-specific tolerance, and that near/missing labels cannot receive
training weight.  It reads one parquet file at a time to keep the full eICU
rebuild auditable on a laptop.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from eicu_raw_measurement_time_augment import _interpolation_max_gap_hours


FILE_RE = re.compile(r"^eicu_(?P<module>.+)_transitions_(?P<horizon>\d+)h\.parquet$")


def _anchor_hours(frame: pd.DataFrame) -> np.ndarray:
    onset = pd.to_numeric(frame.get("onset_hour", 0.0), errors="coerce")
    elapsed = pd.to_numeric(frame.get("hours_since_onset", 0.0), errors="coerce")
    return (onset.fillna(0.0) + elapsed.fillna(0.0)).to_numpy(dtype=np.float64)


def _summary(values: np.ndarray) -> dict[str, float | None]:
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": None, "q10": None, "q90": None, "min": None, "max": None}
    q10, median, q90 = np.quantile(values, [0.10, 0.50, 0.90])
    return {
        "median": float(median),
        "q10": float(q10),
        "q90": float(q90),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def audit_file(path: Path) -> list[dict]:
    match = FILE_RE.match(path.name)
    if match is None:
        return []
    module = match.group("module")
    horizon = int(match.group("horizon"))
    header = pq.ParquetFile(path).schema.names
    quality_columns = sorted(
        column for column in header if column.startswith("future_label_quality_")
    )
    if not quality_columns:
        return [{
            "module": module,
            "horizon_hours": horizon,
            "file": str(path.resolve()),
            "status": "missing_quality_columns",
        }]

    variables = [column.removeprefix("future_label_quality_") for column in quality_columns]
    required = ["onset_hour", "hours_since_onset"]
    for variable in variables:
        required.extend([
            f"future_event_time_hr_{variable}",
            f"future_delta_t_hr_{variable}",
            f"future_label_quality_{variable}",
            f"future_label_weight_{variable}",
            f"future_label_tolerance_strict_hr_{variable}",
            f"future_label_tolerance_near_hr_{variable}",
            f"future_label_left_event_time_hr_{variable}",
            f"future_label_right_event_time_hr_{variable}",
            f"future_label_interpolation_gap_hr_{variable}",
            f"{variable}_t",
        ])
    columns = [column for column in dict.fromkeys(required) if column in header]
    frame = pd.read_parquet(path, columns=columns, engine="pyarrow")
    anchors = _anchor_hours(frame)
    records = []
    for variable in variables:
        quality = frame[f"future_label_quality_{variable}"].astype("string")
        event_time = pd.to_numeric(
            frame[f"future_event_time_hr_{variable}"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        delta = pd.to_numeric(
            frame[f"future_delta_t_hr_{variable}"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        weight = pd.to_numeric(
            frame[f"future_label_weight_{variable}"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        strict = pd.to_numeric(
            frame[f"future_label_tolerance_strict_hr_{variable}"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        near = pd.to_numeric(
            frame[f"future_label_tolerance_near_hr_{variable}"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        current = pd.to_numeric(frame[f"{variable}_t"], errors="coerce").to_numpy(dtype=np.float64)
        exact = quality.eq("exact").to_numpy()
        interpolated = quality.eq("interpolated_dense").to_numpy()
        near_label = quality.eq("near").to_numpy()
        missing = quality.eq("missing").to_numpy() | quality.isna().to_numpy()
        left = pd.to_numeric(
            frame.get(
                f"future_label_left_event_time_hr_{variable}",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        right = pd.to_numeric(
            frame.get(
                f"future_label_right_event_time_hr_{variable}",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        interpolation_gap = pd.to_numeric(
            frame.get(
                f"future_label_interpolation_gap_hr_{variable}",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        finite_event = np.isfinite(event_time)
        finite_delta = np.isfinite(delta)
        strict_timing_ok = exact & finite_delta & np.isfinite(strict) & (np.abs(delta - horizon) <= strict + 1e-6)
        near_timing_ok = (exact | near_label) & finite_delta & np.isfinite(near) & (np.abs(delta - horizon) <= near + 1e-6)
        interpolation_max_gap = _interpolation_max_gap_hours(variable, horizon)
        interpolation_ok = interpolated & (
            np.isfinite(left)
            & np.isfinite(right)
            & np.isfinite(interpolation_gap)
            & (left > anchors + 1e-6)
            & (right > left)
            & (np.abs(interpolation_gap - (right - left)) <= 1e-6)
            & (interpolation_gap <= interpolation_max_gap + 1e-6)
            & finite_delta
            & (np.abs(delta - horizon) <= 1e-6)
        )
        post_anchor_ok = (
            ((exact | near_label) & finite_event & (event_time > anchors + 1e-6))
            | (interpolated & np.isfinite(left) & np.isfinite(right) & (left > anchors + 1e-6))
        )
        positive_delta_ok = (exact | near_label | interpolated) & finite_delta & (delta > 0.0)
        weight_ok = np.where(
            exact,
            weight == 1.0,
            np.where(interpolated, weight == 0.50, weight == 0.0),
        )
        paired_exact = exact & np.isfinite(current) & finite_event
        records.append({
            "module": module,
            "horizon_hours": horizon,
            "variable": variable,
            "file": str(path.resolve()),
            "rows": int(len(frame)),
            "exact": int(exact.sum()),
            "interpolated": int(interpolated.sum()),
            "near": int(near_label.sum()),
            "missing": int(missing.sum()),
            "exact_rate": float(exact.mean()),
            "strict_timing_ok": int(strict_timing_ok.sum()),
            "near_timing_ok": int(near_timing_ok.sum()),
            "interpolation_ok": int(interpolation_ok.sum()),
            "post_anchor_ok": int(post_anchor_ok.sum()),
            "positive_delta_ok": int(positive_delta_ok.sum()),
            "weight_consistent": int(weight_ok.sum()),
            "paired_exact_current_and_future": int(paired_exact.sum()),
            "delta_t_hours": _summary(delta[exact]),
            "label_contract_pass": bool(
                int(strict_timing_ok.sum()) == int(exact.sum())
                and int(interpolation_ok.sum()) == int(interpolated.sum())
                and int(post_anchor_ok.sum()) == int((exact | near_label | interpolated).sum())
                and int(positive_delta_ok.sum()) == int((exact | near_label | interpolated).sum())
                and int(weight_ok.sum()) == len(frame)
            ),
        })
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.data_root.glob("eicu_*_transitions_*h.parquet"))
    if not paths:
        raise FileNotFoundError(f"No transition parquet files under {args.data_root}")
    records = []
    for path in paths:
        records.extend(audit_file(path))
        print(json.dumps({"file": path.name, "records": len(records)}), flush=True)

    valid_records = [record for record in records if "variable" in record]
    aggregate = {}
    for record in valid_records:
        key = (record["module"], record["horizon_hours"], record["variable"])
        item = aggregate.setdefault(key, {
            "module": record["module"],
            "horizon_hours": record["horizon_hours"],
            "variable": record["variable"],
            "rows": 0,
            "exact": 0,
            "interpolated": 0,
            "near": 0,
            "missing": 0,
            "strict_timing_ok": 0,
            "near_timing_ok": 0,
            "interpolation_ok": 0,
            "post_anchor_ok": 0,
            "positive_delta_ok": 0,
            "weight_consistent": 0,
            "paired_exact_current_and_future": 0,
        })
        for field in (
            "rows", "exact", "interpolated", "near", "missing", "strict_timing_ok", "near_timing_ok", "interpolation_ok",
            "post_anchor_ok", "positive_delta_ok", "weight_consistent",
            "paired_exact_current_and_future",
        ):
            item[field] += record[field]
    for item in aggregate.values():
        item["exact_rate"] = item["exact"] / max(item["rows"], 1)
        item["label_contract_pass"] = (
            item["strict_timing_ok"] == item["exact"]
            and item["interpolation_ok"] == item["interpolated"]
            and item["post_anchor_ok"] == item["exact"] + item["interpolated"] + item["near"]
            and item["positive_delta_ok"] == item["exact"] + item["interpolated"] + item["near"]
            and item["weight_consistent"] == item["rows"]
        )

    output = {
        "schema": "whole_body_future_label_quality_audit.v1",
        "data_root": str(args.data_root.resolve()),
        "files": len(paths),
        "target_horizon_records": len(valid_records),
        "aggregate": list(aggregate.values()),
        "all_contracts_pass": bool(
            valid_records and all(item["label_contract_pass"] for item in aggregate.values())
        ),
        "training_policy": {
            "exact_weight": 1.0,
            "interpolated_dense_weight": 0.50,
            "near_weight": 0.0,
            "missing_weight": 0.0,
            "near_labels_are_audit_only": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "files": len(paths),
        "target_horizon_records": len(valid_records),
        "all_contracts_pass": output["all_contracts_pass"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
