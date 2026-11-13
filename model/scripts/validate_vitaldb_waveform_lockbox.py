#!/usr/bin/env python3
"""Evaluate a frozen MIMIC raw-waveform JEPA on an untouched VitalDB lockbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.train_waveform_event_jepa import _ece, _paired_ci, stable_seed
from whole_body_jepa.raw_waveform_jepa import RawWaveformEventJEPA


def _table(frame: pd.DataFrame, artifact: dict[str, object]) -> np.ndarray:
    columns = list(artifact["table_columns"])
    values = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    median = np.asarray(artifact["table_median"], dtype=float)
    scale = np.asarray(artifact["table_scale"], dtype=float)
    missing = ~np.isfinite(values)
    normalized = np.clip((np.where(missing, median, values) - median) / scale, -12.0, 12.0)
    return np.concatenate((normalized, missing.astype(float)), axis=1).astype(np.float32)


def _probability(
    model: RawWaveformEventJEPA,
    temperature: float,
    table: np.ndarray,
    past: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(
            torch.from_numpy(table),
            torch.from_numpy(past),
            torch.from_numpy(mask),
        )["logit"].numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -30.0, 30.0)))


def run(args: argparse.Namespace) -> dict[str, object]:
    artifact = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    frame = pd.read_parquet(args.cohort)
    cache = np.load(args.raw_cache)
    row_index = cache["row_index"].astype(int)
    frame = frame.iloc[row_index].reset_index(drop=True)
    table = _table(frame, artifact)
    past = cache["past"].astype(np.float32)
    mask = cache["past_mask"].astype(np.float32)
    labels = frame["event"].to_numpy(np.float32)
    subjects = frame["subject_id"].astype(str).to_numpy()
    model = RawWaveformEventJEPA(
        table.shape[1], scale_count=len(artifact["past_scales_seconds"])
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    full = _probability(model, float(artifact["temperature"]), table, past, mask)
    masked = _probability(
        model,
        float(artifact["temperature"]),
        table,
        np.zeros_like(past),
        np.zeros_like(mask),
    )
    rng = np.random.default_rng(20260827)
    unique = np.unique(subjects)
    shuffled_subjects = unique.copy()
    rng.shuffle(shuffled_subjects)
    source = {subject: np.flatnonzero(subjects == shuffled) for subject, shuffled in zip(unique, shuffled_subjects)}
    permutation = np.arange(len(subjects))
    for subject in unique:
        target_rows = np.flatnonzero(subjects == subject)
        permutation[target_rows] = np.resize(source[subject], len(target_rows))
    shuffled = _probability(
        model,
        float(artifact["temperature"]),
        table,
        past[permutation],
        mask[permutation],
    )
    probability = {"full": full, "raw_masked": masked, "raw_shuffled": shuffled}
    prevalence = float(labels.mean())
    metrics = {
        name: {
            "brier": float(brier_score_loss(labels, values)),
            "auc": float(roc_auc_score(labels, values)) if len(np.unique(labels)) > 1 else None,
            "ece": float(_ece(labels, values)),
        }
        for name, values in probability.items()
    }
    metrics["external_climatology"] = {
        "brier": float(brier_score_loss(labels, np.full(len(labels), prevalence))),
        "auc": 0.5,
        "ece": 0.0,
    }
    comparisons = {
        "full_vs_raw_masked": _paired_ci(
            labels,
            full,
            masked,
            subjects,
            seed=stable_seed("vitaldb", "masked"),
            repeats=args.bootstrap_repeats,
        ),
        "full_vs_raw_shuffled": _paired_ci(
            labels,
            full,
            shuffled,
            subjects,
            seed=stable_seed("vitaldb", "shuffled"),
            repeats=args.bootstrap_repeats,
        ),
    }
    support_pass = bool(frame["subject_id"].nunique() >= 200 and frame["event"].sum() >= 40)
    external_pass = bool(
        support_pass
        and comparisons["full_vs_raw_masked"]["pass"]
        and comparisons["full_vs_raw_shuffled"]["pass"]
        and metrics["full"]["ece"] <= 0.05
    )
    runtime_eligible = bool(artifact.get("internal_gate_pass", False) and external_pass)
    report = {
        "schema": "whole_body_jepa.vitaldb_waveform_external_validation.v1",
        "checkpoint": str(args.checkpoint),
        "rows": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()),
        "events": int(frame["event"].sum()),
        "prevalence": prevalence,
        "probability_summary": {
            name: {
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values)),
            }
            for name, values in probability.items()
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "support_pass": support_pass,
        "checkpoint_internal_gate_pass": bool(artifact.get("internal_gate_pass", False)),
        "external_gate_pass": external_pass,
        "runtime_eligible": runtime_eligible,
        "lockbox_policy": "frozen_checkpoint_no_vitaldb_tuning",
        "domain": "intraoperative_surgical_not_icu",
        "causal_claim_allowed": False,
        "clinical_decision_allowed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
