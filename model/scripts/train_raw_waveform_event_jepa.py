#!/usr/bin/env python3
"""Gate raw multi-scale Waveform JEPA against table and summary baselines."""

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

from scripts.train_waveform_event_jepa import (
    MatrixTransform,
    PATIENT_SEEDS,
    _columns,
    _ece,
    _fit_temperature,
    _paired_ci,
    _patient_permutation,
    _split_fit_calibration,
    stable_seed,
)
from whole_body_jepa.raw_waveform_jepa import DenseEventModel, RawWaveformEventJEPA


def _augment(
    values: torch.Tensor, mask: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.clone()
    mask = mask.clone()
    drop = torch.rand(mask.shape, generator=generator) < 0.10
    mask[drop] = 0.0
    width = values.shape[-1]
    temporal_width = max(8, width // 8)
    starts = torch.randint(
        0, width - temporal_width + 1, (values.shape[0],), generator=generator
    )
    for row, start in enumerate(starts.tolist()):
        values[row, :, :, start : start + temporal_width] = 0.0
    return values, mask


def _fit_dense(
    values: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    calibration: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> tuple[DenseEventModel, float]:
    torch.manual_seed(seed)
    model = DenseEventModel(values.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    best_state = None
    best_loss = float("inf")
    stale = 0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = fit.copy()
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), 128):
            batch = order[start : start + 128]
            logits = model(torch.from_numpy(values[batch]))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.from_numpy(labels[batch]).float()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(values[calibration]))
            validation = float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.from_numpy(labels[calibration]).float()
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
        raise RuntimeError("dense event model did not train")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(values[calibration])).numpy()
    return model, _fit_temperature(logits, labels[calibration])


def _fit_raw(
    table: np.ndarray,
    past: np.ndarray,
    past_mask: np.ndarray,
    future: np.ndarray,
    future_mask: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    calibration: np.ndarray,
    *,
    seed: int,
    pretrain_epochs: int,
    event_epochs: int,
) -> tuple[RawWaveformEventJEPA, float]:
    torch.manual_seed(seed)
    model = RawWaveformEventJEPA(table.shape[1], scale_count=past.shape[1])
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
            past_batch, mask_batch = _augment(
                torch.from_numpy(past[batch]),
                torch.from_numpy(past_mask[batch]),
                generator,
            )
            output = model(
                torch.from_numpy(table[batch]),
                past_batch,
                mask_batch,
                torch.from_numpy(future[batch]),
                torch.from_numpy(future_mask[batch]),
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
    for _ in range(event_epochs):
        order = fit.copy()
        rng.shuffle(order)
        model.train()
        for start in range(0, len(order), 128):
            batch = order[start : start + 128]
            output = model(
                torch.from_numpy(table[batch]),
                torch.from_numpy(past[batch]),
                torch.from_numpy(past_mask[batch]),
                torch.from_numpy(future[batch]),
                torch.from_numpy(future_mask[batch]),
            )
            loss = model.event_loss(
                output, torch.from_numpy(labels[batch]).float(), jepa_weight=0.10
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.update_target_encoder()
        model.eval()
        with torch.no_grad():
            output = model(
                torch.from_numpy(table[calibration]),
                torch.from_numpy(past[calibration]),
                torch.from_numpy(past_mask[calibration]),
                torch.from_numpy(future[calibration]),
                torch.from_numpy(future_mask[calibration]),
            )
            validation = float(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    output["logit"], torch.from_numpy(labels[calibration]).float()
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
        raise RuntimeError("raw waveform JEPA did not train")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(table[calibration]),
            torch.from_numpy(past[calibration]),
            torch.from_numpy(past_mask[calibration]),
        )["logit"].numpy()
    return model, _fit_temperature(logits, labels[calibration])


def _dense_probability(
    model: DenseEventModel, temperature: float, values: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(values[rows])).numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -30, 30)))


def _raw_probability(
    model: RawWaveformEventJEPA,
    temperature: float,
    table: np.ndarray,
    past: np.ndarray,
    past_mask: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(
            torch.from_numpy(table[rows]),
            torch.from_numpy(past[rows]),
            torch.from_numpy(past_mask[rows]),
        )["logit"].numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -30, 30)))


def _partition_permutation(
    subjects: np.ndarray, partitions: tuple[np.ndarray, ...], seed: int
) -> np.ndarray:
    output = np.arange(len(subjects))
    for partition_index, partition in enumerate(partitions):
        local = np.asarray(partition, dtype=int)
        permutation = _patient_permutation(
            subjects[local], stable_seed(seed, partition_index)
        )
        output[local] = local[permutation]
    return output


def _careunit_split(
    subjects: np.ndarray, careunits: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    heldout = sorted(
        value
        for value in np.unique(careunits)
        if stable_seed("raw_careunit", value) % 3 == 0
    )
    test = np.isin(careunits, heldout)
    test_subjects = np.unique(subjects[test])
    train = ~np.isin(subjects, test_subjects)
    if not train.any() or not test.any():
        raise RuntimeError("care-unit split has no patient-disjoint train or test rows")
    return train, test, heldout


def _forward_time_split(
    subjects: np.ndarray, anchor_time: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    subject_time = (
        pd.DataFrame(
            {
                "subject": subjects,
                "anchor_time": pd.to_datetime(anchor_time).to_numpy(),
            }
        )
        .groupby("subject", sort=False)["anchor_time"]
        .min()
        .sort_values(kind="stable")
    )
    split = int(round(0.70 * len(subject_time)))
    split = min(max(split, 1), len(subject_time) - 1)
    test_subjects = subject_time.index.to_numpy()[split:]
    test = np.isin(subjects, test_subjects)
    train = ~test
    return train, test


def _split_report(
    frame: pd.DataFrame,
    raw: dict[str, np.ndarray],
    raw_table: np.ndarray,
    raw_summary: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    seed: int,
    pretrain_epochs: int,
    event_epochs: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit, calibration = _split_fit_calibration(subjects, train_mask, seed)
    test = np.flatnonzero(test_mask)
    table_transform = MatrixTransform().fit(raw_table[fit])
    summary_transform = MatrixTransform().fit(raw_summary[fit])
    table = table_transform.transform(raw_table)
    summary = summary_transform.transform(raw_summary)
    table_summary = np.concatenate((table, summary), axis=1).astype(np.float32)

    table_model, table_temperature = _fit_dense(
        table, labels, fit, calibration, seed=stable_seed(seed, "table"), epochs=event_epochs
    )
    summary_model, summary_temperature = _fit_dense(
        table_summary,
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "summary"),
        epochs=event_epochs,
    )
    raw_model, raw_temperature = _fit_raw(
        table,
        raw["past"],
        raw["past_mask"],
        raw["future"],
        raw["future_mask"],
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "raw"),
        pretrain_epochs=pretrain_epochs,
        event_epochs=event_epochs,
    )

    permutation = _partition_permutation(
        subjects, (fit, calibration, test), stable_seed(seed, "placebo")
    )
    placebo_model, placebo_temperature = _fit_raw(
        table,
        raw["past"][permutation],
        raw["past_mask"][permutation],
        raw["future"][permutation],
        raw["future_mask"][permutation],
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
        "raw_jepa": _raw_probability(
            raw_model, raw_temperature, table, raw["past"], raw["past_mask"], test
        ),
        "shuffled_raw_placebo": _raw_probability(
            placebo_model,
            placebo_temperature,
            table,
            raw["past"][permutation],
            raw["past_mask"][permutation],
            test,
        ),
    }
    truth = labels[test]
    test_subjects = subjects[test]
    report = {
        "rows": int(len(test)),
        "patients": int(len(np.unique(test_subjects))),
        "prevalence": float(truth.mean()),
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
        "raw_vs_table": _paired_ci(
            truth,
            probability["raw_jepa"],
            probability["table"],
            test_subjects,
            seed=stable_seed(seed, "raw_table"),
            repeats=repeats,
        ),
        "raw_vs_summary": _paired_ci(
            truth,
            probability["raw_jepa"],
            probability["summary"],
            test_subjects,
            seed=stable_seed(seed, "raw_summary"),
            repeats=repeats,
        ),
        "raw_vs_placebo": _paired_ci(
            truth,
            probability["raw_jepa"],
            probability["shuffled_raw_placebo"],
            test_subjects,
            seed=stable_seed(seed, "raw_placebo"),
            repeats=repeats,
        ),
    }
    state = {
        "state_dict": raw_model.state_dict(),
        "temperature": raw_temperature,
        "table_median": table_transform.median,
        "table_scale": table_transform.scale,
    }
    return report, state


def run(args: argparse.Namespace) -> dict[str, Any]:
    full_frame = pd.read_parquet(args.cohort)
    cache = np.load(args.raw_cache)
    row_index = cache["row_index"].astype(int)
    frame = full_frame.iloc[row_index].reset_index(drop=True)
    raw = {
        name: cache[name].astype(np.float32)
        for name in ("past", "future", "past_mask", "future_mask")
    }
    numeric, added, summary_columns, _ = _columns(frame)
    table_columns = numeric + added
    raw_table = frame[table_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    raw_summary = frame[summary_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    labels = frame["event"].to_numpy(np.float32)
    subjects = frame["subject_id"].astype(str).to_numpy()

    patient_reports = []
    for seed in PATIENT_SEEDS:
        train = np.asarray(
            [stable_seed("raw_outer", seed, subject) % 4 != 0 for subject in subjects]
        )
        report, _ = _split_report(
            frame,
            raw,
            raw_table,
            raw_summary,
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
        patient_reports.append(report)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "raw_vs_table": report["raw_vs_table"],
                    "raw_vs_summary": report["raw_vs_summary"],
                    "raw_vs_placebo": report["raw_vs_placebo"],
                }
            ),
            flush=True,
        )

    careunits = frame["careunit"].astype(str).to_numpy()
    careunit_train, careunit_test, heldout_careunits = _careunit_split(
        subjects, careunits
    )
    careunit_report, _ = _split_report(
        frame,
        raw,
        raw_table,
        raw_summary,
        labels,
        subjects,
        careunit_train,
        careunit_test,
        seed=9001,
        pretrain_epochs=args.pretrain_epochs,
        event_epochs=args.event_epochs,
        repeats=args.bootstrap_repeats,
    )
    careunit_report["heldout_careunits"] = sorted(heldout_careunits)

    time_train, time_test = _forward_time_split(subjects, frame["anchor_time"])
    time_report, _ = _split_report(
        frame,
        raw,
        raw_table,
        raw_summary,
        labels,
        subjects,
        time_train,
        time_test,
        seed=9901,
        pretrain_epochs=args.pretrain_epochs,
        event_epochs=args.event_epochs,
        repeats=args.bootstrap_repeats,
    )

    internal_pass = bool(
        all(
            item["raw_vs_table"]["pass"]
            and item["raw_vs_summary"]["pass"]
            and item["raw_vs_placebo"]["pass"]
            for item in patient_reports
        )
        and all(
            item["raw_vs_table"]["pass"]
            and item["raw_vs_summary"]["pass"]
            and item["raw_vs_placebo"]["pass"]
            and item["ece"]["raw_jepa"] <= 0.05
            for item in (careunit_report, time_report)
        )
    )

    all_train = np.ones(len(frame), dtype=bool)
    fit, calibration = _split_fit_calibration(subjects, all_train, 20260826)
    table_transform = MatrixTransform().fit(raw_table[fit])
    table = table_transform.transform(raw_table)
    final_model, final_temperature = _fit_raw(
        table,
        raw["past"],
        raw["past_mask"],
        raw["future"],
        raw["future_mask"],
        labels,
        fit,
        calibration,
        seed=20260826,
        pretrain_epochs=args.pretrain_epochs,
        event_epochs=args.event_epochs,
    )
    artifact = {
        "schema": "whole_body_jepa.raw_waveform_event_jepa.v1",
        "event": args.event_name,
        "table_columns": table_columns,
        "signal_groups": ["ecg", "pleth", "respiration", "arterial_pressure"],
        "past_scales_seconds": [int(value) for value in args.past_scales.split(",")],
        "points_per_scale": int(raw["past"].shape[-1]),
        "state_dict": final_model.state_dict(),
        "temperature": final_temperature,
        "table_median": table_transform.median,
        "table_scale": table_transform.scale,
        "internal_gate_pass": internal_pass,
        "runtime_eligible": False,
        "runtime_blocker": "independent_external_hospital_waveform_lockbox_required",
        "causal_claim_allowed": False,
        "clinical_decision_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    report = {
        "schema": "whole_body_jepa.raw_waveform_event_jepa_audit.v1",
        "event": args.event_name,
        "rows": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()),
        "patient_splits": patient_reports,
        "careunit_holdout": careunit_report,
        "forward_time_holdout": time_report,
        "internal_gate_pass": internal_pass,
        "runtime_eligible": False,
        "external_waveform_lockbox_complete": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"internal_gate_pass": internal_pass}, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--event-epochs", type=int, default=24)
    parser.add_argument("--past-scales", default="60,300,900")
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
