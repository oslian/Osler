"""Observation-aware feature views and component-specific diagnostic gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .change_evaluation import estimate, interval_score
from .prospective_change import recall_at_fpr


def observation_features(frame, scaler, variables, target):
    """Return raw physiology and explicit observation-state features.

    The target's own anchor value is excluded because target history already
    carries it. Missing values remain zero only in the value block; mask and age
    make missing, stale, and freshly measured values distinguishable.
    """

    current, mask, ages = scaler.normalize_current(frame)
    return observation_feature_blocks(current, mask, ages, variables, target)


def observation_feature_blocks(current, mask, ages, variables, target):
    keep = np.asarray([variable != target for variable in variables], dtype=bool)
    raw = current[:, keep]
    observed = np.concatenate([raw, mask[:, keep], ages[:, keep]], axis=1)
    return raw.astype(np.float32), observed.astype(np.float32)


def paired_delta(frame, candidate, reference, truth, *, repetitions=1000):
    delta = np.abs(np.asarray(candidate) - truth) - np.abs(np.asarray(reference) - truth)
    return estimate(frame, delta, repetitions=repetitions)


def point_gate(frame, predictions, truth, current, candidate, references, threshold, *, repetitions=1000):
    comparisons = {
        name: paired_delta(frame, predictions[candidate]["point"], predictions[name]["point"], truth,
                           repetitions=repetitions)
        for name in references
    }
    stable = np.abs(truth - current) < threshold
    stable_harm = paired_delta(
        frame.loc[stable].reset_index(drop=True),
        predictions[candidate]["point"][stable], current[stable], truth[stable],
        repetitions=repetitions,
    )
    passed = all(item["ci95"] is not None and item["ci95"][1] < 0 for item in comparisons.values())
    passed &= stable_harm["ci95"] is not None and stable_harm["ci95"][1] <= 0.05 * threshold
    return {"comparisons": comparisons, "stable_harm_vs_persistence": stable_harm, "pass": bool(passed)}


def risk_gate(frame, predictions, labels, candidate, references, frozen_cutoff, *, repetitions=1000):
    candidate_recall = recall_at_fpr(labels, predictions[candidate]["risk"], frame.subject_id)
    comparisons = {}
    rng = np.random.default_rng(20260908)
    for name in references:
        reference_recall = recall_at_fpr(labels, predictions[name]["risk"], frame.subject_id)
        draws = []
        for _ in range(repetitions):
            rows = rng.integers(len(frame), size=len(frame))
            sampled = labels[rows]
            if len(np.unique(sampled)) < 2:
                continue
            draw_ids = np.arange(len(rows))
            direct = recall_at_fpr(sampled, predictions[candidate]["risk"][rows], draw_ids)
            baseline = recall_at_fpr(sampled, predictions[name]["risk"][rows], draw_ids)
            draws.append(direct - baseline)
        comparisons[name] = {
            "difference": None if candidate_recall is None or reference_recall is None else float(candidate_recall - reference_recall),
            "ci95": None if not draws else np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    alert = predictions[candidate]["risk"] > frozen_cutoff
    fpr = estimate(frame, np.where(~labels, alert, np.nan), repetitions=repetitions)
    passed = fpr["mean"] is not None and fpr["mean"] <= 0.05
    passed &= all(item["ci95"] is not None and item["ci95"][0] > 0 for item in comparisons.values())
    return {"recall_at_matched_5pct_fpr": candidate_recall, "comparisons": comparisons,
            "fpr_at_frozen_source_cutoff": fpr, "pass": bool(passed)}


def interval_gate(frame, predictions, truth, candidate, references, threshold, *, repetitions=1000):
    selected = predictions[candidate]
    valid = np.isfinite(selected["lower"]) & np.isfinite(selected["upper"])
    coverage = estimate(frame, np.where(valid, (selected["lower"] <= truth) & (truth <= selected["upper"]), np.nan),
                        repetitions=repetitions)
    width = estimate(frame, np.where(valid, selected["upper"] - selected["lower"], np.nan), repetitions=repetitions)
    candidate_score = interval_score(truth, selected["lower"], selected["upper"])
    score_comparisons = {}
    for name in references:
        ref = predictions[name]
        ref_valid = np.isfinite(ref["lower"]) & np.isfinite(ref["upper"])
        both = valid & ref_valid
        values = np.full(len(frame), np.nan)
        values[both] = candidate_score[both] - interval_score(truth[both], ref["lower"][both], ref["upper"][both])
        score_comparisons[name] = estimate(frame, values, repetitions=repetitions)
    point_noninferiority = paired_delta(frame, selected["point"], predictions[references[0]]["point"], truth,
                                        repetitions=repetitions)
    unit_coverage = {}
    for unit, rows in frame.groupby("careunit").indices.items():
        if len(rows) >= 30 and str(unit).lower() not in {"unknown", "nan", "none", ""}:
            rows = np.asarray(rows)
            unit_coverage[str(unit)] = estimate(frame.iloc[rows].reset_index(drop=True),
                np.where(valid[rows], (selected["lower"][rows] <= truth[rows]) & (truth[rows] <= selected["upper"][rows]), np.nan),
                repetitions=repetitions)
    passed = coverage["mean"] is not None and 0.87 <= coverage["mean"] <= 0.93
    passed &= point_noninferiority["ci95"] is not None and point_noninferiority["ci95"][1] <= 0.02 * threshold
    passed &= all(item["ci95"] is not None and item["ci95"][1] < 0 for item in score_comparisons.values())
    passed &= bool(unit_coverage) and all(item["mean"] is not None and 0.87 <= item["mean"] <= 0.93
                                         for item in unit_coverage.values())
    return {"coverage": coverage, "width": width, "interval_score_comparisons": score_comparisons,
            "point_noninferiority_vs_history": point_noninferiority, "careunit_coverage": unit_coverage,
            "pass": bool(passed)}
