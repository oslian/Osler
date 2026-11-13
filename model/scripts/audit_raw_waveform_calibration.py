#!/usr/bin/env python3
"""Test patient-weighted calibration without changing the raw-waveform backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.train_raw_waveform_event_jepa import (
    _fit_dense,
    _fit_raw,
    _partition_permutation,
)
from scripts.train_waveform_event_jepa import (
    MatrixTransform,
    PATIENT_SEEDS,
    _ece,
    _paired_ci,
    _split_fit_calibration,
    stable_seed,
)
from whole_body_jepa.probability_calibration import fit_probability_calibrator


METHODS = ("temperature", "platt", "beta", "isotonic")


def _dense_logits(model: torch.nn.Module, values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(values[rows])).numpy()


def _raw_logits(
    model: torch.nn.Module,
    table: np.ndarray,
    past: np.ndarray,
    mask: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        return model(
            torch.from_numpy(table[rows]),
            torch.from_numpy(past[rows]),
            torch.from_numpy(mask[rows]),
        )["logit"].numpy()


def _evaluate_method(
    method: str,
    calibration_logits: dict[str, np.ndarray],
    test_logits: dict[str, np.ndarray],
    calibration_labels: np.ndarray,
    test_labels: np.ndarray,
    calibration_subjects: np.ndarray,
    test_subjects: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> dict[str, object]:
    probability = {}
    for name in calibration_logits:
        calibrator = fit_probability_calibrator(
            method,
            calibration_logits[name],
            calibration_labels,
            calibration_subjects,
        )
        probability[name] = calibrator.predict(test_logits[name])
    full = probability["full"]
    comparisons = {
        f"full_vs_{name}": _paired_ci(
            test_labels,
            full,
            probability[name],
            test_subjects,
            seed=stable_seed(seed, method, name),
            repeats=repeats,
        )
        for name in ("table", "masked", "placebo")
    }
    return {
        "brier": {
            name: float(brier_score_loss(test_labels, values))
            for name, values in probability.items()
        },
        "auc": {
            name: float(roc_auc_score(test_labels, values))
            for name, values in probability.items()
        },
        "ece": {name: float(_ece(test_labels, values)) for name, values in probability.items()},
        "comparisons": comparisons,
        "gate_pass": bool(
            all(item["pass"] for item in comparisons.values())
            and _ece(test_labels, full) <= 0.05
        ),
    }


def _split(
    raw_table: np.ndarray,
    raw: dict[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    seed: int,
    pretrain_epochs: int,
    event_epochs: int,
    repeats: int,
) -> dict[str, object]:
    fit, calibration = _split_fit_calibration(subjects, train, seed)
    test_rows = np.flatnonzero(test)
    transform = MatrixTransform().fit(raw_table[fit])
    table = transform.transform(raw_table)
    table_model, _ = _fit_dense(
        table,
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "calibration_table"),
        epochs=event_epochs,
    )
    full_model, _ = _fit_raw(
        table,
        raw["past"],
        raw["past_mask"],
        raw["future"],
        raw["future_mask"],
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "calibration_full"),
        pretrain_epochs=pretrain_epochs,
        event_epochs=event_epochs,
    )
    permutation = _partition_permutation(
        subjects, (fit, calibration, test_rows), stable_seed(seed, "calibration_placebo")
    )
    placebo_model, _ = _fit_raw(
        table,
        raw["past"][permutation],
        raw["past_mask"][permutation],
        raw["future"][permutation],
        raw["future_mask"][permutation],
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "calibration_placebo_model"),
        pretrain_epochs=pretrain_epochs,
        event_epochs=event_epochs,
    )
    zero_past = np.zeros_like(raw["past"])
    zero_mask = np.zeros_like(raw["past_mask"])
    calibration_logits = {
        "full": _raw_logits(full_model, table, raw["past"], raw["past_mask"], calibration),
        "table": _dense_logits(table_model, table, calibration),
        "masked": _raw_logits(full_model, table, zero_past, zero_mask, calibration),
        "placebo": _raw_logits(
            placebo_model,
            table,
            raw["past"][permutation],
            raw["past_mask"][permutation],
            calibration,
        ),
    }
    test_logits = {
        "full": _raw_logits(full_model, table, raw["past"], raw["past_mask"], test_rows),
        "table": _dense_logits(table_model, table, test_rows),
        "masked": _raw_logits(full_model, table, zero_past, zero_mask, test_rows),
        "placebo": _raw_logits(
            placebo_model,
            table,
            raw["past"][permutation],
            raw["past_mask"][permutation],
            test_rows,
        ),
    }
    return {
        "rows": int(len(test_rows)),
        "patients": int(len(np.unique(subjects[test_rows]))),
        "prevalence": float(labels[test_rows].mean()),
        "methods": {
            method: _evaluate_method(
                method,
                calibration_logits,
                test_logits,
                labels[calibration],
                labels[test_rows],
                subjects[calibration],
                subjects[test_rows],
                seed=seed,
                repeats=repeats,
            )
            for method in METHODS
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    reference = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)
    full_frame = pd.read_parquet(args.cohort)
    cache = np.load(args.raw_cache)
    row_index = cache["row_index"].astype(int)
    frame = full_frame.iloc[row_index].reset_index(drop=True)
    raw = {
        name: cache[name].astype(np.float32)
        for name in ("past", "future", "past_mask", "future_mask")
    }
    table_columns = list(reference["table_columns"])
    raw_table = (
        frame.reindex(columns=table_columns)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    labels = frame["event"].to_numpy(np.float32)
    subjects = frame["subject_id"].astype(str).to_numpy()
    splits = []
    for seed in PATIENT_SEEDS:
        train = np.asarray(
            [stable_seed("calibration_outer", seed, subject) % 4 != 0 for subject in subjects]
        )
        split = _split(
            raw_table,
            raw,
            labels,
            subjects,
            train,
            ~train,
            seed=seed,
            pretrain_epochs=args.pretrain_epochs,
            event_epochs=args.event_epochs,
            repeats=args.bootstrap_repeats,
        )
        split["seed"] = seed
        splits.append(split)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "gate_pass": {
                        method: split["methods"][method]["gate_pass"]
                        for method in METHODS
                    },
                }
            ),
            flush=True,
        )
    pass_counts = {
        method: int(sum(split["methods"][method]["gate_pass"] for split in splits))
        for method in METHODS
    }
    comparator_pass_counts = {
        method: {
            comparator: int(
                sum(
                    split["methods"][method]["comparisons"][comparator]["pass"]
                    for split in splits
                )
            )
            for comparator in ("full_vs_table", "full_vs_masked", "full_vs_placebo")
        }
        for method in METHODS
    }
    report = {
        "schema": "whole_body_jepa.raw_waveform_calibration_audit.v1",
        "rows": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()),
        "methods": list(METHODS),
        "patient_equal_calibration": True,
        "splits": splits,
        "pass_counts": pass_counts,
        "comparator_pass_counts": comparator_pass_counts,
        "patient_gate_pass": [method for method, count in pass_counts.items() if count == 7],
        "candidate_only": True,
        "causal_claim_allowed": False,
        "clinical_decision_allowed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pass_counts": pass_counts, "comparator_pass_counts": comparator_pass_counts}, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--event-epochs", type=int, default=24)
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
