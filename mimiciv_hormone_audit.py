"""Audit newly augmented MIMIC-IV hormone targets.

This evaluates hormone labs as observed factual targets, not as direct hidden
ADH/insulin/PTH state labels. It reuses the existing MIMIC nowcast/forecast
gates and keeps the report aggregate-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eicu_full_variable_coverage_audit import (
    forecast_feature_columns,
    forecast_rows,
    nowcast_feature_columns,
    nowcast_rows,
)
from mimiciv_cross_database_coverage_audit import (
    add_time_holdout_column,
    audit_target_mimic,
    capped_subject_rows,
    stable_seed,
)


DEFAULT_TARGETS = ("cortisol", "pth", "tpo_antibody", "tsh", "total_t4", "free_t4")


def audit(
    cohort: Path,
    targets: tuple[str, ...],
    seeds: tuple[int, ...],
    bootstrap_samples: int,
    min_pairs: int,
    min_subjects: int,
    max_rows_per_target: int,
) -> dict[str, object]:
    frame = add_time_holdout_column(pd.read_parquet(cohort).reset_index(drop=True), 0.67)
    reports: dict[str, object] = {}
    for target in targets:
        print(f"Auditing hormone target: {target}", flush=True)
        target_report: dict[str, object] = {}
        current_rows = nowcast_rows(frame, target)
        if len(current_rows):
            rows, sampling = capped_subject_rows(
                current_rows,
                max_rows_per_target,
                seed=stable_seed("mimiciv-hormone", "nowcast", target),
            )
            target_report["nowcast"] = audit_target_mimic(
                frame,
                target,
                rows,
                nowcast_feature_columns(frame, target),
                seeds,
                0.67,
                min_pairs,
                min_subjects,
                3,
                10.0,
                bootstrap_samples,
                "nowcast",
                0.9,
                100,
                50,
            )
            target_report["nowcast"]["sampling"] = sampling
        future_rows = forecast_rows(frame, target, "tp6")
        if len(future_rows):
            rows, sampling = capped_subject_rows(
                future_rows,
                max_rows_per_target,
                seed=stable_seed("mimiciv-hormone", "forecast", target),
            )
            target_report["forecast"] = audit_target_mimic(
                frame,
                target,
                rows,
                forecast_feature_columns(frame, target),
                seeds,
                0.67,
                min_pairs,
                min_subjects,
                3,
                10.0,
                bootstrap_samples,
                "forecast",
                0.9,
                100,
                50,
            )
            target_report["forecast"]["sampling"] = sampling
        reports[target] = target_report
    return {
        "artifact": "MIMIC-IV hormone target factual observation audit",
        "cohort": str(cohort),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "targets": list(targets),
        "reports": reports,
        "safety_boundary": {
            "direct_hormone_concentration_claim_allowed": False,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "runtime_decision_authority": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--seeds", default="7,11,19,23,37,53,71")
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--min-subjects", type=int, default=20)
    parser.add_argument("--max-rows-per-target", type=int, default=30000)
    args = parser.parse_args()
    report = audit(
        args.cohort,
        tuple(item.strip() for item in args.targets.split(",") if item.strip()),
        tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip()),
        args.bootstrap_samples,
        args.min_pairs,
        args.min_subjects,
        args.max_rows_per_target,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "targets": report["targets"],
        "safety_boundary": report["safety_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
