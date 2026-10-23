"""Seven-seed patient-held-out audit for the JEPA belief adapter candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from dka_world_model import STATE_KEYS, load_checkpoint
from real_world_improvement import (
    ADAPTED_STATE_INDICES,
    arrays,
    fit_ridge,
    predict_ridge,
)
from jepa_belief_adapter_candidate import build_examples
from train_intervention_jepa import choose_device


SEEDS = (7, 11, 19, 23, 37, 53, 71)


def _bootstrap_subject_delta(groups, delta, seed, samples=500):
    unique = np.unique(groups)
    if len(unique) < 5:
        return None
    rng = np.random.default_rng(seed)
    by_subject = np.asarray([
        np.mean(delta[groups == subject]) for subject in unique
    ], dtype=np.float64)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draws[index] = rng.choice(by_subject, size=len(by_subject), replace=True).mean()
    return [round(float(np.quantile(draws, 0.025)), 6), round(float(np.quantile(draws, 0.975)), 6)]


def _split(groups, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    cut = min(max(1, int(round(0.30 * len(unique)))), max(1, len(unique) - 1))
    test_groups = set(unique[:cut].tolist())
    test = np.asarray([group in test_groups for group in groups], dtype=bool)
    return ~test, test


def _one_seed(data, seed):
    train_mask, test_mask = _split(data["groups"], seed)
    train = np.flatnonzero(train_mask)
    test = np.flatnonzero(test_mask)
    result = {}
    for target_index in ADAPTED_STATE_INDICES:
        target_name = STATE_KEYS[target_index]
        fit = fit_ridge(data, train, target_index, "combined", 10.0)
        if fit is None:
            continue
        predicted = predict_ridge(fit, data, test, target_index)
        truth = data["target"][test, target_index]
        current = data["current"][test, target_index]
        base = data["base"][test, target_index]
        valid = np.isfinite(truth) & np.isfinite(current) & np.isfinite(predicted)
        if valid.sum() < 20:
            continue
        groups = data["groups"][test][valid]
        candidate_error = np.abs(predicted[valid] - truth[valid])
        persistence_error = np.abs(current[valid] - truth[valid])
        base_error = np.abs(base[valid] - truth[valid])
        delta = candidate_error - persistence_error
        ci = _bootstrap_subject_delta(groups, delta, seed + target_index)
        result[target_name] = {
            "n": int(valid.sum()),
            "subjects": int(len(np.unique(groups))),
            "candidate_mae": round(float(mean_absolute_error(truth[valid], predicted[valid])), 4),
            "persistence_mae": round(float(mean_absolute_error(truth[valid], current[valid])), 4),
            "base_jepa_mae": round(float(mean_absolute_error(truth[valid], base[valid])), 4),
            "point_delta_vs_persistence": round(float(delta.mean()), 6),
            "bootstrap_95_ci_delta": ci,
            "significant_vs_persistence": bool(ci is not None and ci[1] < 0.0),
            "beats_base_jepa": bool(
                mean_absolute_error(truth[valid], predicted[valid])
                < mean_absolute_error(truth[valid], base[valid])
            ),
        }
    return result


def _aggregate(reports):
    output = {}
    for target in sorted({name for report in reports for name in report}):
        rows = [report[target] for report in reports if target in report]
        output[target] = {
            "seeds_evaluated": len(rows),
            "significant_vs_persistence": sum(
                row["significant_vs_persistence"] for row in rows
            ),
            "beats_base_jepa": sum(row["beats_base_jepa"] for row in rows),
            "mean_candidate_mae": round(float(np.mean([row["candidate_mae"] for row in rows])), 4),
            "mean_persistence_mae": round(float(np.mean([row["persistence_mae"] for row in rows])), 4),
            "mean_base_jepa_mae": round(float(np.mean([row["base_jepa_mae"] for row in rows])), 4),
            "validated_patient_gate": bool(
                len(rows) == len(SEEDS)
                and all(row["significant_vs_persistence"] for row in rows)
                and all(row["beats_base_jepa"] for row in rows)
            ),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="dka_symbolic_jepa_v5.pt")
    parser.add_argument("--mimic", default="dka_transitions_6h_mimiciv_full_v31_icd.parquet")
    parser.add_argument("--output", default="jepa_belief_adapter_robustness.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--include-beliefs",
        action="store_true",
        help="reproduce the earlier belief-adapter ablation; disabled by default",
    )
    args = parser.parse_args()
    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    frame = pd.read_parquet(args.mimic)
    examples = build_examples(
        model,
        frame,
        device,
        include_beliefs=args.include_beliefs,
    )
    data = arrays(examples)
    reports = [_one_seed(data, seed) for seed in SEEDS]
    report = {
        "test_type": "JEPA belief adapter seven-seed patient-held-out audit",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "cohort": str(Path(args.mimic).resolve()),
        "examples": len(examples),
        "stays": int(len(np.unique(data["groups"]))),
        "seeds": list(SEEDS),
        "belief_features_used": bool(args.include_beliefs),
        "per_seed": {str(seed): result for seed, result in zip(SEEDS, reports)},
        "aggregate": _aggregate(reports),
        "hospital_holdout": {
            "status": "not_run",
            "reason": "the transition artifact has no hospital or care-unit identifier",
        },
        "boundary": {
            "factual_only": True,
            "causal_claims_allowed": False,
            "checkpoint_promotion_allowed": False,
            "adapter_is_candidate_only": True,
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nsaved report: {args.output}")


if __name__ == "__main__":
    main()
