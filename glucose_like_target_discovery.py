"""Discover targets with glucose-like observation and forecast dynamics.

Target selection is intentionally patient-disjoint from subsequent model
training and evaluation.  The discovery report is aggregate-only; subject IDs
are written only to the local evaluation parquet boundary and never to JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whole_body_joint_jepa import load_joint_event_examples


def _safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 20:
        return np.nan
    left = left[finite]
    right = right[finite]
    if float(np.std(left)) < 1.0e-8 or float(np.std(right)) < 1.0e-8:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _robust_scale(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale < 1.0e-6:
        scale = float(np.std(values))
    return max(scale, 1.0e-6)


def _skewness(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return np.nan
    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    if scale < 1.0e-8:
        return 0.0
    return float(np.mean((centered / scale) ** 3))


def target_descriptor(frame, target, horizons):
    current = pd.to_numeric(frame.get(f"{target}_t"), errors="coerce")
    observed = frame.get(f"_observed_{target}")
    if observed is None:
        observed = current.notna()
    observed = pd.Series(observed, index=frame.index).fillna(False).astype(bool)
    age = pd.to_numeric(frame.get(f"{target}_age_hr"), errors="coerce")
    scale = _robust_scale(current)
    horizon_column = pd.to_numeric(frame["horizon_hours"], errors="coerce")
    descriptor = {
        "anchor_observed_rate": float(observed.mean()),
        "anchor_available_rate": float(current.notna().mean()),
        "median_measurement_age_log": float(
            np.log1p(age[observed & age.notna()].median())
        )
        if bool((observed & age.notna()).any())
        else np.nan,
    }
    all_delta = []
    all_current = []
    all_future = []
    total_future = 0
    future_subjects = set()
    for horizon in horizons:
        selected_horizon = horizon_column == float(horizon)
        future = pd.to_numeric(frame.get(f"future_{target}"), errors="coerce")
        pair = selected_horizon & current.notna() & future.notna()
        rows = int(pair.sum())
        total_future += rows
        if rows:
            future_subjects.update(
                frame.loc[pair, "subject_id"].astype(str).unique().tolist()
            )
        delta = (future[pair] - current[pair]).to_numpy(dtype=np.float64)
        normalized_delta = delta / scale
        all_delta.extend(normalized_delta.tolist())
        all_current.extend(current[pair].to_numpy(dtype=np.float64).tolist())
        all_future.extend(future[pair].to_numpy(dtype=np.float64).tolist())
        descriptor[f"future_support_{int(horizon)}h_log"] = float(
            np.log1p(rows)
        )
        descriptor[f"median_abs_delta_{int(horizon)}h"] = (
            float(np.median(np.abs(normalized_delta))) if rows else np.nan
        )
        descriptor[f"changed_fraction_{int(horizon)}h"] = (
            float(np.mean(np.abs(normalized_delta) >= 0.10))
            if rows
            else np.nan
        )
        descriptor[f"persistence_corr_{int(horizon)}h"] = (
            _safe_correlation(current[pair], future[pair]) if rows else np.nan
        )
        actual_delta = pd.to_numeric(
            frame.get(f"future_delta_t_hr_{target}"), errors="coerce"
        )
        timing = actual_delta[pair]
        descriptor[f"label_jitter_{int(horizon)}h"] = (
            float(np.median(np.abs(timing - float(horizon))) / max(horizon, 1))
            if bool(timing.notna().any())
            else np.nan
        )
    all_delta = np.asarray(all_delta, dtype=np.float64)
    descriptor.update(
        {
            "future_rows_log": float(np.log1p(total_future)),
            "future_subjects_log": float(np.log1p(len(future_subjects))),
            "median_abs_delta_all": (
                float(np.median(np.abs(all_delta))) if len(all_delta) else np.nan
            ),
            "q90_abs_delta_all": (
                float(np.quantile(np.abs(all_delta), 0.90))
                if len(all_delta)
                else np.nan
            ),
            "mean_signed_delta_all": (
                float(np.mean(all_delta)) if len(all_delta) else np.nan
            ),
            "delta_skewness": _skewness(all_delta),
            "current_future_correlation": _safe_correlation(
                all_current, all_future
            ),
        }
    )
    weight = pd.to_numeric(
        frame.get(f"future_label_weight_{target}"), errors="coerce"
    )
    descriptor["mean_future_label_weight"] = (
        float(weight.mean()) if bool(weight.notna().any()) else np.nan
    )
    return descriptor, {
        "future_rows": int(total_future),
        "future_subjects": int(len(future_subjects)),
    }


def rank_glucose_like_targets(descriptors, support, *, top_k=12):
    names = list(descriptors)
    feature_names = sorted(
        {feature for values in descriptors.values() for feature in values}
    )
    matrix = np.asarray(
        [
            [descriptors[name].get(feature, np.nan) for feature in feature_names]
            for name in names
        ],
        dtype=np.float64,
    )
    center = np.nanmedian(matrix, axis=0)
    center[~np.isfinite(center)] = 0.0
    filled = np.where(np.isfinite(matrix), matrix, center)
    q25 = np.quantile(filled, 0.25, axis=0)
    q75 = np.quantile(filled, 0.75, axis=0)
    scale = q75 - q25
    scale[scale < 1.0e-6] = 1.0
    standardized = np.clip((filled - center) / scale, -8.0, 8.0)
    if "glucose" not in names:
        raise ValueError("glucose is not available in the canonical schema")
    glucose_index = names.index("glucose")
    distances = np.sqrt(
        np.mean((standardized - standardized[glucose_index]) ** 2, axis=1)
    )
    eligible = [
        index
        for index, name in enumerate(names)
        if support[name]["future_rows"] >= 300
        and support[name]["future_subjects"] >= 40
    ]
    ranked = sorted(eligible, key=lambda index: (distances[index], names[index]))
    selected_indices = ranked[: max(1, int(top_k))]
    if glucose_index not in selected_indices:
        selected_indices = [glucose_index, *selected_indices[:-1]]
    selected = [names[index] for index in selected_indices]
    ranking = [
        {
            "target": names[index],
            "distance_to_glucose": float(distances[index]),
            **support[names[index]],
            "selected": bool(index in selected_indices),
        }
        for index in ranked
    ]
    return selected, ranking, feature_names


def run(args):
    modules = tuple(value.strip() for value in args.modules.split(",") if value.strip())
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    frame, variables, module_names = load_joint_event_examples(
        args.event_examples,
        modules,
        horizons,
        args.max_stays,
        min_modules=args.min_modules,
        organ_contract_mode="canonical_organs",
        history_mode="multiscale",
        history_lags_hours=(48, 24, 12, 8, 6, 4, 3, 2, 1, 0),
    )
    subjects = np.asarray(sorted(frame["subject_id"].astype(str).unique()))
    rng = np.random.default_rng(args.seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    discovery_count = max(50, int(round(len(shuffled) * args.discovery_fraction)))
    discovery_subjects = set(shuffled[:discovery_count].tolist())
    discovery_mask = frame["subject_id"].astype(str).isin(discovery_subjects)
    discovery = frame.loc[discovery_mask].copy()
    evaluation = frame.loc[~discovery_mask].copy()
    descriptors = {}
    support = {}
    for target in variables:
        descriptors[target], support[target] = target_descriptor(
            discovery, target, horizons
        )
    selected, ranking, feature_names = rank_glucose_like_targets(
        descriptors, support, top_k=args.top_k
    )
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_parquet(args.evaluation_output, index=False)
    source = pd.read_parquet(args.event_examples)
    used_subjects = set(frame["subject_id"].astype(str).unique().tolist())
    external_holdout = source.loc[
        ~source["subject_id"].astype(str).isin(used_subjects)
    ].copy()
    args.external_holdout_output.parent.mkdir(parents=True, exist_ok=True)
    external_holdout.to_parquet(args.external_holdout_output, index=False)
    report = {
        "schema": "glucose_like_target_discovery.v1",
        "status": "discovery_only_target_selection",
        "source_rows": int(len(frame)),
        "source_subjects": int(len(subjects)),
        "discovery_rows": int(len(discovery)),
        "discovery_subjects": int(len(discovery_subjects)),
        "evaluation_rows": int(len(evaluation)),
        "evaluation_subjects": int(evaluation["subject_id"].nunique()),
        "external_holdout_rows": int(len(external_holdout)),
        "external_holdout_subjects": int(
            external_holdout["subject_id"].nunique()
        ),
        "patient_disjoint_discovery_and_evaluation": True,
        "patient_disjoint_external_holdout": True,
        "selection_seed": int(args.seed),
        "selection_features": feature_names,
        "selected_targets": selected,
        "ranking": ranking,
        "evaluation_event_examples": str(args.evaluation_output.resolve()),
        "external_holdout_event_examples": str(
            args.external_holdout_output.resolve()
        ),
        "row_level_identifiers_in_report": False,
        "causal_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-examples", type=Path, required=True)
    parser.add_argument("--modules", required=True)
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--min-modules", type=int, default=2)
    parser.add_argument("--discovery-fraction", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--external-holdout-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "evaluation_output": str(args.evaluation_output.resolve()),
                "external_holdout_output": str(
                    args.external_holdout_output.resolve()
                ),
                "selected_targets": report["selected_targets"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
