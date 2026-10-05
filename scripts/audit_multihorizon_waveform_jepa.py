#!/usr/bin/env python3
"""Audit whether ordered event horizons stabilize raw Waveform JEPA."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.train_raw_waveform_event_jepa import (
    _augment,
    _careunit_split,
    _dense_probability,
    _fit_dense,
    _forward_time_split,
    _partition_permutation,
)
from scripts.train_waveform_event_jepa import (
    MatrixTransform,
    PATIENT_SEEDS,
    _columns,
    _ece,
    _fit_temperature,
    _paired_ci,
    _split_fit_calibration,
    stable_seed,
)
from whole_body_jepa.raw_waveform_jepa import MultiHorizonWaveformJEPA


HORIZONS = (5, 10, 15, 30)
KEYS = ("subject_id", "stay_id", "record", "anchor_time", "anchor_sample")


def _labels(base: pd.DataFrame, directory: Path) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(base[list(KEYS)])
    output = []
    for minutes in HORIZONS:
        frame = pd.read_parquet(
            directory / f"mimiciv_multimodal_hypoxemia_{minutes}m.parquet"
        )
        series = pd.Series(
            frame["event"].to_numpy(float),
            index=pd.MultiIndex.from_frame(frame[list(KEYS)]),
        )
        output.append(series.reindex(keys).to_numpy(float))
    return np.column_stack(output).astype(np.float32)


def _fit_raw(
    table: np.ndarray,
    raw: dict[str, np.ndarray],
    labels: np.ndarray,
    fit: np.ndarray,
    calibration: np.ndarray,
    *,
    seed: int,
    pretrain_epochs: int,
    event_epochs: int,
) -> tuple[MultiHorizonWaveformJEPA, float]:
    torch.manual_seed(seed)
    model = MultiHorizonWaveformJEPA(
        table.shape[1],
        horizon_count=len(HORIZONS),
        scale_count=raw["past"].shape[1],
    )
    pretrain_optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-3,
        weight_decay=2e-4,
    )
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(pretrain_epochs):
        order = fit.copy()
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), 128):
            batch = order[start : start + 128]
            past, mask = _augment(
                torch.from_numpy(raw["past"][batch]),
                torch.from_numpy(raw["past_mask"][batch]),
                generator,
            )
            output = model(
                torch.from_numpy(table[batch]),
                past,
                mask,
                torch.from_numpy(raw["future"][batch]),
                torch.from_numpy(raw["future_mask"][batch]),
            )
            loss = model.jepa_loss(output)
            pretrain_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            pretrain_optimizer.step()
            model.update_target_encoder()

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-4)
    best_state = None
    best_loss = float("inf")
    stale = 0
    label_values = np.nan_to_num(labels, nan=0.0).astype(np.float32)
    label_mask = np.isfinite(labels).astype(np.float32)
    for _ in range(event_epochs):
        order = fit.copy()
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), 128):
            batch = order[start : start + 128]
            output = model(
                torch.from_numpy(table[batch]),
                torch.from_numpy(raw["past"][batch]),
                torch.from_numpy(raw["past_mask"][batch]),
                torch.from_numpy(raw["future"][batch]),
                torch.from_numpy(raw["future_mask"][batch]),
            )
            loss = model.multihorizon_loss(
                output,
                torch.from_numpy(label_values[batch]),
                torch.from_numpy(label_mask[batch]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.update_target_encoder()
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.from_numpy(table[calibration]),
                torch.from_numpy(raw["past"][calibration]),
                torch.from_numpy(raw["past_mask"][calibration]),
            )["logit"][:, 2]
            validation = float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.from_numpy(label_values[calibration, 2])
                )
            )
        if validation < best_loss - 1e-4:
            best_loss = validation
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is None:
        raise RuntimeError("multi-horizon model did not train")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(table[calibration]),
            torch.from_numpy(raw["past"][calibration]),
            torch.from_numpy(raw["past_mask"][calibration]),
        )["logit"][:, 2].numpy()
    return model, _fit_temperature(logits, label_values[calibration, 2])


def _probability(
    model: MultiHorizonWaveformJEPA,
    temperature: float,
    table: np.ndarray,
    raw: dict[str, np.ndarray],
    rows: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(
            torch.from_numpy(table[rows]),
            torch.from_numpy(raw["past"][rows]),
            torch.from_numpy(raw["past_mask"][rows]),
        )["logit"][:, 2].numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -30.0, 30.0)))


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
    pretrain_epochs: int,
    event_epochs: int,
    repeats: int,
) -> dict[str, Any]:
    fit, calibration = _split_fit_calibration(subjects, train_mask, seed)
    test = np.flatnonzero(test_mask)
    table = MatrixTransform().fit(raw_table[fit]).transform(raw_table)
    summary = MatrixTransform().fit(raw_summary[fit]).transform(raw_summary)
    table_summary = np.concatenate((table, summary), axis=1).astype(np.float32)
    target = labels[:, 2]
    table_model, table_temperature = _fit_dense(
        table, target, fit, calibration, seed=stable_seed(seed, "table"), epochs=event_epochs
    )
    summary_model, summary_temperature = _fit_dense(
        table_summary,
        target,
        fit,
        calibration,
        seed=stable_seed(seed, "summary"),
        epochs=event_epochs,
    )
    candidate, candidate_temperature = _fit_raw(
        table,
        raw,
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "multi_horizon"),
        pretrain_epochs=pretrain_epochs,
        event_epochs=event_epochs,
    )
    permutation = _partition_permutation(
        subjects, (fit, calibration, test), stable_seed(seed, "placebo")
    )
    placebo_raw = {name: values[permutation] for name, values in raw.items()}
    placebo, placebo_temperature = _fit_raw(
        table,
        placebo_raw,
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "placebo_model"),
        pretrain_epochs=pretrain_epochs,
        event_epochs=event_epochs,
    )
    probability = {
        "table": _dense_probability(table_model, table_temperature, table, test),
        "summary": _dense_probability(
            summary_model, summary_temperature, table_summary, test
        ),
        "multi_horizon": _probability(
            candidate, candidate_temperature, table, raw, test
        ),
        "shuffled_placebo": _probability(
            placebo, placebo_temperature, table, placebo_raw, test
        ),
    }
    truth = target[test]
    test_subjects = subjects[test]
    comparisons = {
        name: _paired_ci(
            truth,
            probability["multi_horizon"],
            probability[comparator],
            test_subjects,
            seed=stable_seed(seed, name),
            repeats=repeats,
        )
        for name, comparator in (
            ("vs_table", "table"),
            ("vs_summary", "summary"),
            ("vs_placebo", "shuffled_placebo"),
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
            and _ece(truth, probability["multi_horizon"]) <= 0.05
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = pd.read_parquet(args.cohort)
    cache = np.load(args.raw_cache)
    frame = base.iloc[cache["row_index"].astype(int)].reset_index(drop=True)
    labels = _labels(base, args.horizon_directory)[cache["row_index"].astype(int)]
    raw = {
        name: cache[name].astype(np.float32)
        for name in ("past", "past_mask", "future", "future_mask")
    }
    numeric, added, summary_columns, _ = _columns(frame)
    raw_table = frame[numeric + added].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    raw_summary = frame[summary_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    subjects = frame["subject_id"].astype(str).to_numpy()
    reports = []
    for seed in PATIENT_SEEDS:
        train = np.asarray(
            [stable_seed("multihorizon_outer", seed, subject) % 4 != 0 for subject in subjects]
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
            pretrain_epochs=args.pretrain_epochs,
            event_epochs=args.event_epochs,
            repeats=args.bootstrap_repeats,
        )
        report["seed"] = seed
        reports.append(report)
        print(json.dumps({"seed": seed, "gate_pass": report["gate_pass"]}), flush=True)
    careunits = frame["careunit"].astype(str).to_numpy()
    train, test, heldout = _careunit_split(subjects, careunits)
    careunit = _split_report(
        raw_table, raw_summary, raw, labels, subjects, train, test,
        seed=9001, pretrain_epochs=args.pretrain_epochs,
        event_epochs=args.event_epochs, repeats=args.bootstrap_repeats,
    )
    careunit["heldout_careunits"] = heldout
    train, test = _forward_time_split(subjects, frame["anchor_time"])
    forward_time = _split_report(
        raw_table, raw_summary, raw, labels, subjects, train, test,
        seed=9901, pretrain_epochs=args.pretrain_epochs,
        event_epochs=args.event_epochs, repeats=args.bootstrap_repeats,
    )
    internal_pass = bool(
        all(item["gate_pass"] for item in reports)
        and careunit["gate_pass"]
        and forward_time["gate_pass"]
    )
    report = {
        "schema": "whole_body_jepa.multihorizon_waveform_event_audit.v1",
        "horizons_minutes": list(HORIZONS),
        "target_horizon_minutes": 15,
        "rows": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()),
        "patient_splits": reports,
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
    parser.add_argument("--horizon-directory", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--event-epochs", type=int, default=24)
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
