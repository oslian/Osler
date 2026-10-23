"""Audit validated waveform nowcasts as inputs to future forecasts.

The candidate is a two-stage factual pipeline:

    pre-anchor waveform -> current-state nowcast -> future table forecast

Both stages are refit inside each patient/care-unit/time discovery split.  No
future columns are used by the nowcast stage, and the result remains a
candidate forecast increment even when the same-anchor nowcast is validated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eicu_nowcasting_audit import _bootstrap_ci
from eicu_sepsis_target_router import split_subjects
from mimiciv_cross_database_coverage_audit import add_time_holdout_column
from mimiciv_treatment_context_accuracy_audit import fit_ridge_residual
from mimiciv_waveform_nowcast_audit import (
    DEFAULT_SEEDS,
    _finite_target_rows,
    _fit_ridge as fit_nowcast_ridge,
)
from mimiciv_waveform_precision_audit import (
    attach_waveform_features,
    build_segment_index,
    candidate_anchor_rows,
    forecast_feature_columns,
    forecast_rows,
    wave_feature_sets,
)


QUALITY_RULES = {
    "heart_rate": {
        "valid_columns": (
            "wavebeat_ecg_1_quality_valid_fraction",
            "wavebeat_ecg_2_quality_valid_fraction",
            "wavebeat_ecg_3_quality_valid_fraction",
        ),
        "agreement_column": "wavebeat_ecg_consensus_agreement_fraction",
        "lead_column": "wavebeat_ecg_consensus_lead_count",
        "min_valid_fraction": 0.90,
        "min_agreement": 0.75,
        "min_leads": 2.0,
    },
    "respiratory_rate": {
        "valid_columns": ("wavebeat_respiration_1_quality_valid_fraction",),
        "agreement_column": "wavebeat_respiration_consensus_agreement_fraction",
        "lead_column": "wavebeat_respiration_consensus_lead_count",
        "min_valid_fraction": 0.90,
        "min_agreement": 0.75,
        "min_leads": 1.0,
    },
}


def quality_mask(frame: pd.DataFrame, target: str) -> np.ndarray:
    """Return a fixed pre-anchor signal-quality gate for one target."""

    rule = QUALITY_RULES.get(target)
    if rule is None:
        return np.ones(len(frame), dtype=bool)
    values = frame[list(rule["valid_columns"])].apply(pd.to_numeric, errors="coerce")
    valid = values.mean(axis=1).to_numpy(dtype=np.float64) >= float(rule["min_valid_fraction"])
    agreement = pd.to_numeric(frame[rule["agreement_column"]], errors="coerce").to_numpy(dtype=np.float64)
    leads = pd.to_numeric(frame[rule["lead_column"]], errors="coerce").to_numpy(dtype=np.float64)
    return valid & (agreement >= float(rule["min_agreement"])) & (leads >= float(rule["min_leads"]))


def _group_delta(rows: pd.DataFrame, candidate: np.ndarray, comparator: np.ndarray, seed: int) -> dict[str, object]:
    truth = rows["truth"].to_numpy(dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    finite = np.isfinite(truth) & np.isfinite(candidate) & np.isfinite(comparator)
    if not finite.any():
        return {"rows": 0, "subjects": 0, "point_delta": None, "ci": [None, None], "significant": False}
    grouped = pd.DataFrame({
        "subject_id": rows.loc[finite, "subject_id"].to_numpy(dtype=object),
        "delta": np.abs(candidate[finite] - truth[finite]) - np.abs(comparator[finite] - truth[finite]),
    }).groupby("subject_id")["delta"].mean().to_numpy(dtype=np.float64)
    ci = _bootstrap_ci(grouped, seed=seed, samples=300)
    return {
        "rows": int(finite.sum()),
        "subjects": int(len(grouped)),
        "mae_candidate": float(np.abs(candidate[finite] - truth[finite]).mean()),
        "mae_comparator": float(np.abs(comparator[finite] - truth[finite]).mean()),
        "point_delta": float(grouped.mean()),
        "ci": ci,
        "significant": bool(ci and ci[1] < 0.0),
    }


def _fit_two_stage(
    frame: pd.DataFrame,
    target: str,
    future_suffix: str,
    train_groups: set[object],
    heldout_groups: set[object],
    group_column: str,
    seed: int,
) -> dict[str, object]:
    future_rows = forecast_rows(frame, target, future_suffix)
    train_future = future_rows[future_rows[group_column].isin(train_groups)].copy()
    heldout_future = future_rows[future_rows[group_column].isin(heldout_groups)].copy()
    if train_future.empty or heldout_future.empty:
        return {"status": "insufficient_support"}

    nowcast_rows = _finite_target_rows(frame, target)
    train_nowcast = nowcast_rows[nowcast_rows[group_column].isin(train_groups)].copy()
    now_features = wave_feature_sets(frame, target, profile="target_group")["wave"]
    table_nowcast = [
        column for column in frame.columns
        if column.endswith("_t") or column.endswith("_age_hr") or column.startswith("hist_")
    ]
    target_col = f"{target}_t"
    age_col = f"{target}_age_hr"
    table_nowcast = sorted({column for column in table_nowcast if column not in {target_col, age_col} and not column.endswith("_tp1") and not column.endswith("_tp3")})
    nowcast_features = sorted(set(table_nowcast + now_features))
    current_train = fit_nowcast_ridge(
        train_nowcast,
        train_future,
        nowcast_features,
        alpha=5.0,
        seed=seed + 101,
    )
    current_heldout = fit_nowcast_ridge(
        train_nowcast,
        heldout_future,
        nowcast_features,
        alpha=5.0,
        seed=seed + 102,
    )
    completion_column = f"wave_nowcast_{target}"
    train_future[completion_column] = current_train
    heldout_future[completion_column] = current_heldout

    table_forecast = forecast_feature_columns(frame, target)
    candidate_forecast = sorted(set(table_forecast + [completion_column]))
    baseline = fit_ridge_residual(
        train_future,
        heldout_future,
        target,
        table_forecast,
        alpha=5.0,
        seed=seed + 201,
    )
    candidate = fit_ridge_residual(
        train_future,
        heldout_future,
        target,
        candidate_forecast,
        alpha=5.0,
        seed=seed + 202,
    )
    placebo = fit_ridge_residual(
        train_future,
        heldout_future,
        target,
        table_forecast,
        alpha=5.0,
        seed=seed + 203,
        noise_count=1,
    )
    quality = quality_mask(heldout_future, target)
    quality_candidate = np.where(quality, candidate, baseline)
    quality_placebo = np.where(quality, placebo, baseline)
    return {
        "status": "evaluated",
        "candidate_vs_baseline": _group_delta(heldout_future, candidate, baseline, seed + 301),
        "candidate_vs_placebo": _group_delta(heldout_future, candidate, placebo, seed + 302),
        "quality_fallback": {
            "coverage": float(np.mean(quality)) if len(quality) else 0.0,
            "candidate_vs_baseline": _group_delta(
                heldout_future,
                quality_candidate,
                baseline,
                seed + 303,
            ),
            "candidate_vs_placebo": _group_delta(
                heldout_future,
                quality_candidate,
                quality_placebo,
                seed + 304,
            ),
        },
        "support": {
            "train_rows": int(len(train_future)),
            "heldout_rows": int(len(heldout_future)),
            "train_nowcast_rows": int(len(train_nowcast)),
            "nowcast_feature_count": int(len(nowcast_features)),
            "forecast_feature_count": int(len(table_forecast)),
        },
    }


def audit_target(frame: pd.DataFrame, target: str, future_suffix: str) -> dict[str, object]:
    future_rows = forecast_rows(frame, target, future_suffix)
    random_reports = []
    for seed in DEFAULT_SEEDS:
        train, heldout = split_subjects(future_rows, seed, 0.7)
        random_reports.append(_fit_two_stage(frame, target, future_suffix, train, heldout, "subject_id", seed))

    external = {}
    if "first_careunit" in future_rows and future_rows["first_careunit"].nunique(dropna=True) >= 2:
        train, heldout = split_subjects(future_rows, 9001, 0.7, group_column="first_careunit")
        external["careunit"] = _fit_two_stage(frame, target, future_suffix, train, heldout, "first_careunit", 9001)
    if "time_holdout_group" in future_rows:
        external["time"] = _fit_two_stage(frame, target, future_suffix, {"early"}, {"late"}, "time_holdout_group", 9901)

    base_count = sum(bool(x.get("candidate_vs_baseline", {}).get("significant")) for x in random_reports)
    placebo_count = sum(bool(x.get("candidate_vs_placebo", {}).get("significant")) for x in random_reports)
    quality_base_count = sum(
        bool(x.get("quality_fallback", {}).get("candidate_vs_baseline", {}).get("significant"))
        for x in random_reports
    )
    quality_placebo_count = sum(
        bool(x.get("quality_fallback", {}).get("candidate_vs_placebo", {}).get("significant"))
        for x in random_reports
    )
    external_pass = all(
        value.get("candidate_vs_baseline", {}).get("significant")
        and value.get("candidate_vs_placebo", {}).get("significant")
        for value in external.values()
        if value.get("status") == "evaluated"
    )
    return {
        "support": {
            "rows": int(len(future_rows)),
            "subjects": int(future_rows["subject_id"].nunique()) if len(future_rows) else 0,
        },
        "random_patient_splits": {
            "candidate_beats_baseline_splits": int(base_count),
            "candidate_beats_placebo_splits": int(placebo_count),
            "quality_fallback_beats_baseline_splits": int(quality_base_count),
            "quality_fallback_beats_placebo_splits": int(quality_placebo_count),
        },
        "external": external,
        "validated": bool(base_count == len(DEFAULT_SEEDS) and placebo_count == len(DEFAULT_SEEDS) and external_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--waveform-root", type=Path, required=True)
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-hours", type=int, choices=(1, 3), required=True)
    parser.add_argument("--targets", nargs="+", default=["heart_rate", "respiratory_rate"])
    args = parser.parse_args()

    segments = build_segment_index(args.waveform_root, args.mimic_dir)
    transitions = add_time_holdout_column(pd.read_parquet(args.cohort), 0.7)
    anchors = candidate_anchor_rows(transitions, segments, 60.0, 10000, 7105)
    frame, extraction = attach_waveform_features(anchors, 60.0, feature_mode="beat")
    suffix = f"tp{args.horizon_hours}"
    targets = {target: audit_target(frame, target, suffix) for target in args.targets}
    report = {
        "artifact": "MIMIC-IV waveform nowcast-to-forecast audit",
        "horizon_hours": args.horizon_hours,
        "feature_extraction": extraction,
        "targets": targets,
        "boundary": {
            "factual_observed_inputs_only": True,
            "causal_claim_allowed": False,
            "counterfactual_claim_allowed": False,
            "clinical_claim_allowed": False,
            "forecast_promotion_allowed": False,
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({target: value["validated"] for target, value in targets.items()}, indent=2))


if __name__ == "__main__":
    main()
