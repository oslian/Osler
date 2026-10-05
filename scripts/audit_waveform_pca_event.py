#!/usr/bin/env python3
"""Audit low-rank raw-waveform event models with patient-held-out gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.train_raw_waveform_event_jepa import (
    _careunit_split,
    _forward_time_split,
    _partition_permutation,
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
from whole_body_jepa.probability_calibration import (
    fit_probability_calibrator,
    patient_equal_weights,
)


def _wave_matrix(past: np.ndarray, mask: np.ndarray) -> np.ndarray:
    expanded = np.repeat(mask[..., None], past.shape[-1], axis=-1)
    return np.concatenate(
        ((past * expanded).reshape(len(past), -1), mask.reshape(len(mask), -1)),
        axis=1,
    ).astype(np.float32)


def _fit_logistic(
    values: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    fit: np.ndarray,
    c_value: float,
) -> LogisticRegression:
    model = LogisticRegression(C=c_value, max_iter=1000, solver="lbfgs")
    model.fit(
        values[fit],
        labels[fit],
        sample_weight=patient_equal_weights(subjects[fit]),
    )
    return model


def _calibrated_probability(
    model: LogisticRegression,
    values: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    calibrator = fit_probability_calibrator(
        "temperature",
        model.decision_function(values[calibration]),
        labels[calibration],
        subjects[calibration],
    )
    return calibrator.predict(model.decision_function(values[test]))


def _select_dense(
    values: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    fit: np.ndarray,
    calibration: np.ndarray,
) -> tuple[LogisticRegression, float]:
    best: tuple[float, LogisticRegression, float] | None = None
    for c_value in (0.01, 0.1, 1.0):
        model = _fit_logistic(values, labels, subjects, fit, c_value)
        probability = _calibrated_probability(
            model, values, labels, subjects, calibration, calibration
        )
        score = float(brier_score_loss(labels[calibration], probability))
        if best is None or score < best[0]:
            best = (score, model, c_value)
    if best is None:
        raise RuntimeError("dense logistic selection failed")
    return best[1], best[2]


def _fit_low_rank(
    table: np.ndarray,
    wave: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    fit: np.ndarray,
    calibration: np.ndarray,
    *,
    seed: int,
) -> tuple[PCA, LogisticRegression, dict[str, float | int]]:
    best: tuple[float, PCA, LogisticRegression, dict[str, float | int]] | None = None
    for components in (8, 16, 32):
        if components >= min(len(fit), wave.shape[1]):
            continue
        projector = PCA(
            n_components=components,
            whiten=True,
            svd_solver="randomized",
            random_state=seed,
        ).fit(wave[fit])
        projected = projector.transform(wave).astype(np.float32)
        values = np.concatenate((table, projected), axis=1)
        for c_value in (0.01, 0.1, 1.0):
            model = _fit_logistic(values, labels, subjects, fit, c_value)
            probability = _calibrated_probability(
                model, values, labels, subjects, calibration, calibration
            )
            score = float(brier_score_loss(labels[calibration], probability))
            settings = {"components": components, "c": c_value}
            if best is None or score < best[0]:
                best = (score, projector, model, settings)
    if best is None:
        raise RuntimeError("low-rank waveform selection failed")
    return best[1], best[2], best[3]


def _split_report(
    raw_table: np.ndarray,
    raw_summary: np.ndarray,
    wave: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    fit, calibration = _split_fit_calibration(subjects, train_mask, seed)
    test = np.flatnonzero(test_mask)
    table = MatrixTransform().fit(raw_table[fit]).transform(raw_table)
    summary = MatrixTransform().fit(raw_summary[fit]).transform(raw_summary)
    table_summary = np.concatenate((table, summary), axis=1).astype(np.float32)

    table_model, table_c = _select_dense(
        table, labels, subjects, fit, calibration
    )
    summary_model, summary_c = _select_dense(
        table_summary, labels, subjects, fit, calibration
    )
    projector, candidate, settings = _fit_low_rank(
        table,
        wave,
        labels,
        subjects,
        fit,
        calibration,
        seed=stable_seed(seed, "candidate_pca"),
    )
    candidate_values = np.concatenate(
        (table, projector.transform(wave).astype(np.float32)), axis=1
    )

    permutation = _partition_permutation(
        subjects, (fit, calibration, test), stable_seed(seed, "pca_placebo")
    )
    placebo_projector, placebo, placebo_settings = _fit_low_rank(
        table,
        wave[permutation],
        labels,
        subjects,
        fit,
        calibration,
        seed=stable_seed(seed, "placebo_pca"),
    )
    placebo_values = np.concatenate(
        (table, placebo_projector.transform(wave[permutation]).astype(np.float32)),
        axis=1,
    )
    probability = {
        "table": _calibrated_probability(
            table_model, table, labels, subjects, calibration, test
        ),
        "summary": _calibrated_probability(
            summary_model, table_summary, labels, subjects, calibration, test
        ),
        "low_rank_waveform": _calibrated_probability(
            candidate, candidate_values, labels, subjects, calibration, test
        ),
        "shuffled_waveform_placebo": _calibrated_probability(
            placebo, placebo_values, labels, subjects, calibration, test
        ),
    }
    truth = labels[test]
    test_subjects = subjects[test]
    comparisons = {
        "vs_table": _paired_ci(
            truth,
            probability["low_rank_waveform"],
            probability["table"],
            test_subjects,
            seed=stable_seed(seed, "pca_table"),
            repeats=repeats,
        ),
        "vs_summary": _paired_ci(
            truth,
            probability["low_rank_waveform"],
            probability["summary"],
            test_subjects,
            seed=stable_seed(seed, "pca_summary"),
            repeats=repeats,
        ),
        "vs_placebo": _paired_ci(
            truth,
            probability["low_rank_waveform"],
            probability["shuffled_waveform_placebo"],
            test_subjects,
            seed=stable_seed(seed, "pca_placebo_ci"),
            repeats=repeats,
        ),
    }
    return {
        "rows": int(len(test)),
        "patients": int(len(np.unique(test_subjects))),
        "prevalence": float(truth.mean()),
        "selected": settings,
        "placebo_selected": placebo_settings,
        "table_c": table_c,
        "summary_c": summary_c,
        "brier": {
            name: float(brier_score_loss(truth, values))
            for name, values in probability.items()
        },
        "auc": {
            name: float(roc_auc_score(truth, values))
            if len(np.unique(truth)) > 1
            else None
            for name, values in probability.items()
        },
        "ece": {name: _ece(truth, values) for name, values in probability.items()},
        "comparisons": comparisons,
        "gate_pass": bool(
            all(item["pass"] for item in comparisons.values())
            and _ece(truth, probability["low_rank_waveform"]) <= 0.05
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    full_frame = pd.read_parquet(args.cohort)
    cache = np.load(args.raw_cache)
    frame = full_frame.iloc[cache["row_index"].astype(int)].reset_index(drop=True)
    numeric, added, summary_columns, _ = _columns(frame)
    raw_table = frame[numeric + added].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    raw_summary = frame[summary_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    wave = _wave_matrix(
        cache["past"].astype(np.float32), cache["past_mask"].astype(np.float32)
    )
    labels = frame["event"].to_numpy(np.float32)
    subjects = frame["subject_id"].astype(str).to_numpy()

    patient_reports = []
    for seed in PATIENT_SEEDS:
        train = np.asarray(
            [stable_seed("pca_outer", seed, subject) % 4 != 0 for subject in subjects]
        )
        report = _split_report(
            raw_table,
            raw_summary,
            wave,
            labels,
            subjects,
            train,
            ~train,
            seed=seed,
            repeats=args.bootstrap_repeats,
        )
        report["seed"] = seed
        patient_reports.append(report)
        print(json.dumps({"seed": seed, "gate_pass": report["gate_pass"]}), flush=True)

    careunits = frame["careunit"].astype(str).to_numpy()
    train, test, heldout = _careunit_split(subjects, careunits)
    careunit = _split_report(
        raw_table,
        raw_summary,
        wave,
        labels,
        subjects,
        train,
        test,
        seed=9001,
        repeats=args.bootstrap_repeats,
    )
    careunit["heldout_careunits"] = heldout
    train, test = _forward_time_split(subjects, frame["anchor_time"])
    forward_time = _split_report(
        raw_table,
        raw_summary,
        wave,
        labels,
        subjects,
        train,
        test,
        seed=9901,
        repeats=args.bootstrap_repeats,
    )
    internal_pass = bool(
        all(item["gate_pass"] for item in patient_reports)
        and careunit["gate_pass"]
        and forward_time["gate_pass"]
    )
    report = {
        "schema": "whole_body_jepa.low_rank_waveform_event_audit.v1",
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
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
