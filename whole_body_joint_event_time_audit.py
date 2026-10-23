"""Audit future-label support against target-specific raw event-time labels.

The transition tables are regularized prediction grids, while the augmented
eICU tables retain a strictly-after-anchor timestamp for the next raw event of
each target.  This audit keeps those concepts separate and reports whether a
finite future label has at least one raw event-time witness.  It does not use
measurement timing as a physiological feature and never creates row-level
outputs.

The audit is intentionally conservative:

* ``future_without_next_event`` is a hard consistency warning.  A future value
  cannot be trusted as a raw target if no post-anchor event exists.
* ``first_event_within_label_window`` is only a lower-bound diagnostic.  The
  augmented column stores the *first* event after the anchor, not every event
  used by the nearest-to-horizon extractor.  It must not be interpreted as the
  exact timestamp of the future label.
* No target is promoted by this file.  It only supplies evidence for a later
  target-specific label contract.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FILE_RE = re.compile(r"^eicu_(?P<module>.+)_transitions_(?P<horizon>\d+)h\.parquet$")
DEFAULT_HORIZONS = (1, 3, 6, 12, 24, 48)
DEFAULT_FIRST_EVENT_WINDOW_TOLERANCE_HOURS = 2.0
EPSILON_HOURS = 1.0e-6


def transition_files(root: Path) -> Iterable[tuple[str, int, Path]]:
    for path in sorted(root.glob("eicu_*_transitions_*h.parquet")):
        match = FILE_RE.match(path.name)
        if match is None:
            continue
        yield match.group("module"), int(match.group("horizon")), path


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        return list(pd.read_parquet(path, engine=None).columns)


def finite_mask(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").notna()


def audit_file(
    module: str,
    horizon: int,
    path: Path,
    first_event_window_tolerance_hours: float,
) -> dict[str, object]:
    columns = parquet_columns(path)
    time_columns = sorted(
        column
        for column in columns
        if column.startswith("next_measurement_time_hr_")
    )
    targets = [column.removeprefix("next_measurement_time_hr_") for column in time_columns]
    required = ["stay_id"]
    for target in targets:
        required.extend(
            column
            for column in (
                f"{target}_t",
                f"{target}_tp{horizon}",
                f"next_measurement_time_hr_{target}",
            )
            if column in columns
        )
    required = list(dict.fromkeys(required))
    frame = pd.read_parquet(path, columns=required)

    target_reports: list[dict[str, object]] = []
    total_future = 0
    total_future_without_event = 0
    total_current_future = 0
    for target in targets:
        current_column = f"{target}_t"
        future_column = f"{target}_tp{horizon}"
        time_column = f"next_measurement_time_hr_{target}"
        if not {current_column, future_column, time_column}.issubset(frame.columns):
            continue

        current = finite_mask(frame[current_column])
        future = finite_mask(frame[future_column])
        next_time = pd.to_numeric(frame[time_column], errors="coerce")
        next_event = next_time.notna()
        positive_next_event = next_event & (next_time > EPSILON_HOURS)
        first_event_in_label_window = positive_next_event & (
            next_time <= float(horizon) + float(first_event_window_tolerance_hours)
        )
        current_future = current & future
        future_without_event = future & ~next_event
        future_with_event = future & next_event
        future_without_first_event_in_window = future & ~first_event_in_label_window

        n_future = int(future.sum())
        n_future_without_event = int(future_without_event.sum())
        total_future += n_future
        total_future_without_event += n_future_without_event
        total_current_future += int(current_future.sum())

        finite_times = next_time[next_event]
        raw_source_target_supported = bool(next_event.any())
        if not raw_source_target_supported:
            event_time_status = "source_coverage_gap"
        elif n_future_without_event:
            event_time_status = "future_label_without_post_anchor_event"
        else:
            event_time_status = "consistent_with_available_raw_source"

        target_report = {
            "target": target,
            "rows": int(len(frame)),
            "stays": int(frame["stay_id"].nunique()),
            "current_observed": int(current.sum()),
            "future_observed": n_future,
            "current_future_pair": int(current_future.sum()),
            "next_raw_event_observed": int(next_event.sum()),
            "positive_next_raw_event_observed": int(positive_next_event.sum()),
            "future_with_any_next_raw_event": int(future_with_event.sum()),
            "future_without_next_raw_event": n_future_without_event,
            "future_without_next_raw_event_rate": (
                float(n_future_without_event / n_future) if n_future else None
            ),
            "raw_event_source_has_any_target_events": raw_source_target_supported,
            "event_time_audit_status": event_time_status,
            "source_coverage_gap_target_horizon": not raw_source_target_supported,
            "first_next_event_within_horizon_plus_label_tolerance": int(
                first_event_in_label_window.sum()
            ),
            "future_with_first_event_in_horizon_window": int(
                (future & first_event_in_label_window).sum()
            ),
            "future_without_first_event_in_horizon_window": int(
                future_without_first_event_in_window.sum()
            ),
            "first_event_window_tolerance_hours": float(first_event_window_tolerance_hours),
            "extractor_future_tolerance_hours": float(min(2.0, max(0.25, horizon / 2.0))),
            "first_next_event_median_hours": (
                float(finite_times.median()) if len(finite_times) else None
            ),
            "first_next_event_p90_hours": (
                float(finite_times.quantile(0.90)) if len(finite_times) else None
            ),
            "future_label_event_time_consistent": (
                None if not raw_source_target_supported else n_future_without_event == 0
            ),
        }
        target_reports.append(target_report)

    return {
        "module": module,
        "horizon_hours": int(horizon),
        "path": str(path.resolve()),
        "rows": int(len(frame)),
        "stays": int(frame["stay_id"].nunique()),
        "targets": target_reports,
        "target_count": int(len(target_reports)),
        "future_label_count": int(total_future),
        "future_label_without_next_event_count": int(total_future_without_event),
        "current_future_pair_count": int(total_current_future),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmented-root",
        type=Path,
        default=Path("eicu_time_augmented"),
        help="Directory containing eICU augmented transition parquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("whole_body_joint_event_time_audit.json"),
    )
    parser.add_argument(
        "--label-tolerance-hours",
        type=float,
        default=DEFAULT_FIRST_EVENT_WINDOW_TOLERANCE_HOURS,
        dest="first_event_window_tolerance_hours",
        help="Diagnostic window for the first post-anchor event; not the future-label tolerance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = list(transition_files(args.augmented_root))
    if not files:
        raise FileNotFoundError(f"No augmented transition files found under {args.augmented_root}")
    if args.first_event_window_tolerance_hours < 0:
        raise ValueError("--label-tolerance-hours must be non-negative")

    reports = []
    for index, (module, horizon, path) in enumerate(files, start=1):
        report = audit_file(
            module,
            horizon,
            path,
            args.first_event_window_tolerance_hours,
        )
        reports.append(report)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(files)}",
                    "module": module,
                    "horizon_hours": horizon,
                    "rows": report["rows"],
                    "targets": report["target_count"],
                    "future_labels": report["future_label_count"],
                    "future_without_event": report["future_label_without_next_event_count"],
                }
            ),
            flush=True,
        )

    target_reports = [target for report in reports for target in report["targets"]]
    inconsistent = [
        target
        for target in target_reports
        if target["event_time_audit_status"] == "future_label_without_post_anchor_event"
    ]
    source_gaps = [
        target
        for target in target_reports
        if target["event_time_audit_status"] == "source_coverage_gap"
    ]
    summary = {
        "files": len(reports),
        "module_count": len({report["module"] for report in reports}),
        "horizon_count": len({report["horizon_hours"] for report in reports}),
        "target_horizon_tasks": len(target_reports),
        "future_label_count": int(sum(report["future_label_count"] for report in reports)),
        "future_label_without_next_event_count": int(
            sum(report["future_label_without_next_event_count"] for report in reports)
        ),
        "inconsistent_target_horizon_tasks": len(inconsistent),
        "source_coverage_gap_target_horizon_tasks": len(source_gaps),
        "future_label_event_time_consistency": len(inconsistent) == 0,
        "first_event_window_tolerance_hours": float(args.first_event_window_tolerance_hours),
        "extractor_future_tolerance_rule": "min(2h, max(0.25h, horizon/2))",
        "strict_timestamp_rule": "raw event offset strictly greater than anchor",
        "future_value_rule": "nearest raw value within horizon +/- extractor tolerance",
        "first_next_event_caveat": (
            "The stored next-measurement time is the first event after the anchor; "
            "it is not asserted to be the exact event used by the nearest-to-horizon label."
        ),
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
        "promotion_allowed": False,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "whole_body_joint_event_time_audit.v1",
                "augmented_root": str(args.augmented_root.resolve()),
                "summary": summary,
                "files": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output.resolve()), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
