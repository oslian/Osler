#!/usr/bin/env python3
"""Audit a fixed deep ensemble of raw Waveform JEPA models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.train_raw_waveform_event_jepa import (
    _careunit_split,
    _dense_probability,
    _fit_dense,
    _fit_raw,
    _forward_time_split,
    _partition_permutation,
    _raw_probability,
)
from scripts.train_waveform_event_jepa import (
    MatrixTransform,
    PATIENT_SEEDS,
    _columns,
    _ece,
    _paired_ci,
    _split_fit_calibration,
    stable_seed,
)


def _ensemble_probability(
    models: list[tuple[object, float]],
    table: np.ndarray,
    raw: dict[str, np.ndarray],
    rows: np.ndarray,
) -> np.ndarray:
    return np.mean(
        [
            _raw_probability(
                model,
                temperature,
                table,
                raw["past"],
                raw["past_mask"],
                rows,
            )
            for model, temperature in models
        ],
        axis=0,
    )


def _fit_ensemble(
    table: np.ndarray,
    raw: dict[str, np.ndarray],
    labels: np.ndarray,
    fit: np.ndarray,
    calibration: np.ndarray,
    *,
    seed: int,
    ensemble_size: int,
    pretrain_epochs: int,
    event_epochs: int,
) -> list[tuple[object, float]]:
    return [
        _fit_raw(
            table,
            raw["past"],
            raw["past_mask"],
            raw["future"],
            raw["future_mask"],
            labels,
            fit,
            calibration,
            seed=stable_seed(seed, "member", member),
            pretrain_epochs=pretrain_epochs,
            event_epochs=event_epochs,
        )
        for member in range(ensemble_size)
    ]


def _split_report(
    raw_table: np.ndarray,
    raw_summary: np.ndarray,
    raw: dict[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    seed: int,
    ensemble_size: int,
    pretrain_epochs: int,
    event_epochs: int,
    repeats: int,
) -> dict[str, Any]:
    fit, calibration = _split_fit_calibration(subjects, train_mask, seed)
    test = np.flatnonzero(test_mask)
    table = MatrixTransform().fit(raw_table[fit]).transform(raw_table)
    summary = MatrixTransform().fit(raw_summary[fit]).transform(raw_summary)
    table_summary = np.concatenate((table, summary), axis=1).astype(np.float32)
    table_model, table_temperature = _fit_dense(
        table,
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "table"),
        epochs=event_epochs,
    )
    summary_model, summary_temperature = _fit_dense(
        table_summary,
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "summary"),
        epochs=event_epochs,
    )
    candidate = _fit_ensemble(
        table,
        raw,
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "candidate"),
        ensemble_size=ensemble_size,
        pretrain_epochs=pretrain_epochs,
        event_epochs=event_epochs,
    )
    permutation = _partition_permutation(
        subjects, (fit, calibration, test), stable_seed(seed, "placebo")
    )
    placebo_raw = {name: values[permutation] for name, values in raw.items()}
    placebo = _fit_ensemble(
        table,
        placebo_raw,
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "placebo"),
        ensemble_size=ensemble_size,
        pretrain_epochs=pretrain_epochs,
        event_epochs=event_epochs,
    )
    probability = {
        "table": _dense_probability(table_model, table_temperature, table, test),
        "summary": _dense_probability(
            summary_model, summary_temperature, table_summary, test
        ),
        "deep_ensemble": _ensemble_probability(candidate, table, raw, test),
        "shuffled_deep_ensemble": _ensemble_probability(
            placebo, table, placebo_raw, test
        ),
    }
    truth = labels[test]
    test_subjects = subjects[test]
    comparisons = {
        name: _paired_ci(
            truth,
            probability["deep_ensemble"],
            probability[comparator],
            test_subjects,
            seed=stable_seed(seed, name),
            repeats=repeats,
        )
        for name, comparator in (
            ("vs_table", "table"),
            ("vs_summary", "summary"),
            ("vs_placebo", "shuffled_deep_ensemble"),
        )
    }
    return {
        "rows": int(len(test)),
        "patients": int(len(np.unique(test_subjects))),
        "brier": {
            name: float(brier_score_loss(truth, values))
            for name, values in probability.items()
        },
        "auc": {
            name: float(roc_auc_score(truth, values))
            for name, values in probability.items()
        },
        "ece": {name: _ece(truth, values) for name, values in probability.items()},
        "comparisons": comparisons,
        "gate_pass": bool(
            all(item["pass"] for item in comparisons.values())
            and _ece(truth, probability["deep_ensemble"]) <= 0.05
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    full_frame = pd.read_parquet(args.cohort)
    cache = np.load(args.raw_cache)
    frame = full_frame.iloc[cache["row_index"].astype(int)].reset_index(drop=True)
    raw = {
        name: cache[name].astype(np.float32)
        for name in ("past", "past_mask", "future", "future_mask")
    }
    numeric, added, summary_columns, _ = _columns(frame)
    raw_table = frame[numeric + added].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    raw_summary = frame[summary_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    labels = frame["event"].to_numpy(np.float32)
    subjects = frame["subject_id"].astype(str).to_numpy()
    patient_reports = []
    for seed in PATIENT_SEEDS:
        train = np.asarray(
            [stable_seed("ensemble_outer", seed, subject) % 4 != 0 for subject in subjects]
        )
        report = _split_report(
            raw_table,
            raw_summary,
            raw,
            labels,
            subjects,
            train,
            ~train,
            seed=seed,
            ensemble_size=args.ensemble_size,
            pretrain_epochs=args.pretrain_epochs,
            event_epochs=args.event_epochs,
            repeats=args.bootstrap_repeats,
        )
        report["seed"] = seed
        patient_reports.append(report)
        print(json.dumps({"seed": seed, "gate_pass": report["gate_pass"]}), flush=True)

    careunits = frame["careunit"].astype(str).to_numpy()
    train, test, heldout = _careunit_split(subjects, careunits)
    careunit = _split_report(
        raw_table, raw_summary, raw, labels, subjects, train, test,
        seed=9001, ensemble_size=args.ensemble_size,
        pretrain_epochs=args.pretrain_epochs, event_epochs=args.event_epochs,
        repeats=args.bootstrap_repeats,
    )
    careunit["heldout_careunits"] = heldout
    train, test = _forward_time_split(subjects, frame["anchor_time"])
    forward_time = _split_report(
        raw_table, raw_summary, raw, labels, subjects, train, test,
        seed=9901, ensemble_size=args.ensemble_size,
        pretrain_epochs=args.pretrain_epochs, event_epochs=args.event_epochs,
        repeats=args.bootstrap_repeats,
    )
    internal_pass = bool(
        all(item["gate_pass"] for item in patient_reports)
        and careunit["gate_pass"]
        and forward_time["gate_pass"]
    )
    report = {
        "schema": "whole_body_jepa.raw_waveform_deep_ensemble_audit.v1",
        "ensemble_size": int(args.ensemble_size),
        "rows": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()),
        "patient_splits": patient_reports,
        "careunit_holdout": careunit,
        "forward_time_holdout": forward_time,
        "internal_gate_pass": internal_pass,
        "runtime_eligible": False,
        "causal_claim_allowed": False,
        "clinical_decision_allowed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"internal_gate_pass": internal_pass}, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--event-epochs", type=int, default=24)
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
