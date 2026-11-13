#!/usr/bin/env python3
"""Gate direct future-SpO2 supervision for raw-waveform JEPA event risk."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts.train_raw_waveform_event_jepa import (
    _fit_dense,
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
from whole_body_jepa.raw_waveform_jepa import MultiTaskResidualWaveformJEPA


AUXILIARY_COLUMNS = (
    "future_spo2_min",
    "future_spo2_mean",
    "future_spo2_std",
    "future_low_fraction",
)


def _fit_multitask(
    table: np.ndarray,
    raw: dict[str, np.ndarray],
    labels: np.ndarray,
    auxiliary: np.ndarray,
    fit: np.ndarray,
    calibration: np.ndarray,
    *,
    seed: int,
    auxiliary_weight: float,
    pretrain_epochs: int,
    event_epochs: int,
) -> MultiTaskResidualWaveformJEPA:
    torch.manual_seed(seed)
    model = MultiTaskResidualWaveformJEPA(table.shape[1], auxiliary_dim=auxiliary.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-4)
    rng = np.random.default_rng(seed)
    best_state = None
    best_loss = float("inf")
    stale = 0
    total_epochs = pretrain_epochs + event_epochs
    for epoch in range(total_epochs):
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
            event_weight = 0.0 if epoch < pretrain_epochs else 1.0
            loss = (
                event_weight
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    output["logit"], torch.from_numpy(labels[batch]).float()
                )
                + 0.10 * model.jepa_loss(output)
                + float(auxiliary_weight)
                * torch.nn.functional.smooth_l1_loss(
                    output["physiology_prediction"],
                    torch.from_numpy(auxiliary[batch]),
                )
                + 0.01 * output["residual_logit"].square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.update_target_encoder()
        if epoch < pretrain_epochs:
            continue
        model.eval()
        with torch.no_grad():
            output = model(
                torch.from_numpy(table[calibration]),
                torch.from_numpy(raw["past"][calibration]),
                torch.from_numpy(raw["past_mask"][calibration]),
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
        raise RuntimeError("multi-task waveform JEPA did not train")
    model.load_state_dict(best_state)
    model.eval()
    return model


def _logits(
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


def _dense_logits(model: torch.nn.Module, table: np.ndarray, rows: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(table[rows])).numpy()


def _calibrated(
    calibration_logits: np.ndarray,
    test_logits: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> np.ndarray:
    return fit_probability_calibrator(
        "temperature", calibration_logits, labels, subjects
    ).predict(test_logits)


def _split(
    raw_table: np.ndarray,
    raw_auxiliary: np.ndarray,
    raw: dict[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    seed: int,
    auxiliary_weights: tuple[float, ...],
    pretrain_epochs: int,
    event_epochs: int,
    repeats: int,
) -> dict[str, object]:
    fit, calibration = _split_fit_calibration(subjects, train, seed)
    test_rows = np.flatnonzero(test)
    table_transform = MatrixTransform().fit(raw_table[fit])
    table = table_transform.transform(raw_table)
    auxiliary_transform = MatrixTransform().fit(raw_auxiliary[fit])
    auxiliary = auxiliary_transform.transform(raw_auxiliary, add_missing=False)
    table_model, _ = _fit_dense(
        table,
        labels,
        fit,
        calibration,
        seed=stable_seed(seed, "multitask_table"),
        epochs=event_epochs,
    )
    table_probability = _calibrated(
        _dense_logits(table_model, table, calibration),
        _dense_logits(table_model, table, test_rows),
        labels[calibration],
        subjects[calibration],
    )
    output = {}
    for auxiliary_weight in auxiliary_weights:
        permutation = _partition_permutation(
            subjects,
            (fit, calibration, test_rows),
            stable_seed(seed, "multitask_placebo", auxiliary_weight),
        )
        full_model = _fit_multitask(
            table,
            raw,
            labels,
            auxiliary,
            fit,
            calibration,
            seed=stable_seed(seed, "multitask_full", auxiliary_weight),
            auxiliary_weight=auxiliary_weight,
            pretrain_epochs=pretrain_epochs,
            event_epochs=event_epochs,
        )
        placebo_raw = {
            name: raw[name][permutation]
            for name in ("past", "past_mask", "future", "future_mask")
        }
        placebo_model = _fit_multitask(
            table,
            placebo_raw,
            labels,
            auxiliary,
            fit,
            calibration,
            seed=stable_seed(seed, "multitask_placebo_model", auxiliary_weight),
            auxiliary_weight=auxiliary_weight,
            pretrain_epochs=pretrain_epochs,
            event_epochs=event_epochs,
        )
        zero_past = np.zeros_like(raw["past"])
        zero_mask = np.zeros_like(raw["past_mask"])
        probability = {"table": table_probability}
        for name, model, past, mask in (
            ("full", full_model, raw["past"], raw["past_mask"]),
            ("masked", full_model, zero_past, zero_mask),
            ("placebo", placebo_model, placebo_raw["past"], placebo_raw["past_mask"]),
        ):
            probability[name] = _calibrated(
                _logits(model, table, past, mask, calibration),
                _logits(model, table, past, mask, test_rows),
                labels[calibration],
                subjects[calibration],
            )
        comparisons = {
            f"full_vs_{name}": _paired_ci(
                labels[test_rows],
                probability["full"],
                probability[name],
                subjects[test_rows],
                seed=stable_seed(seed, auxiliary_weight, name),
                repeats=repeats,
            )
            for name in ("table", "masked", "placebo")
        }
        output[str(auxiliary_weight)] = {
            "brier": {
                name: float(brier_score_loss(labels[test_rows], values))
                for name, values in probability.items()
            },
            "auc": {
                name: float(roc_auc_score(labels[test_rows], values))
                for name, values in probability.items()
            },
            "ece": {
                name: float(_ece(labels[test_rows], values))
                for name, values in probability.items()
            },
            "comparisons": comparisons,
            "gate_pass": bool(
                all(item["pass"] for item in comparisons.values())
                and _ece(labels[test_rows], probability["full"]) <= 0.05
            ),
        }
    return {
        "rows": int(len(test_rows)),
        "patients": int(len(np.unique(subjects[test_rows]))),
        "prevalence": float(labels[test_rows].mean()),
        "weights": output,
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
    raw_table = (
        frame.reindex(columns=list(reference["table_columns"]))
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    raw_auxiliary = (
        frame.loc[:, AUXILIARY_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    labels = frame["event"].to_numpy(np.float32)
    subjects = frame["subject_id"].astype(str).to_numpy()
    weights = tuple(float(value) for value in args.auxiliary_weights.split(","))
    splits = []
    for seed in PATIENT_SEEDS:
        train = np.asarray(
            [stable_seed("multitask_outer", seed, subject) % 4 != 0 for subject in subjects]
        )
        split = _split(
            raw_table,
            raw_auxiliary,
            raw,
            labels,
            subjects,
            train,
            ~train,
            seed=seed,
            auxiliary_weights=weights,
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
                        str(weight): split["weights"][str(weight)]["gate_pass"]
                        for weight in weights
                    },
                }
            ),
            flush=True,
        )
    pass_counts = {
        str(weight): int(
            sum(split["weights"][str(weight)]["gate_pass"] for split in splits)
        )
        for weight in weights
    }
    comparator_pass_counts = {
        str(weight): {
            comparator: int(
                sum(
                    split["weights"][str(weight)]["comparisons"][comparator]["pass"]
                    for split in splits
                )
            )
            for comparator in ("full_vs_table", "full_vs_masked", "full_vs_placebo")
        }
        for weight in weights
    }
    report = {
        "schema": "whole_body_jepa.multitask_waveform_jepa_audit.v1",
        "rows": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()),
        "auxiliary_columns": list(AUXILIARY_COLUMNS),
        "auxiliary_weights": list(weights),
        "splits": splits,
        "pass_counts": pass_counts,
        "comparator_pass_counts": comparator_pass_counts,
        "patient_gate_pass": [weight for weight, count in pass_counts.items() if count == 7],
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
    parser.add_argument("--auxiliary-weights", default="0.1,0.3,1.0")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--event-epochs", type=int, default=24)
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
