#!/usr/bin/env python3
"""Train and gate a multimodal Waveform JEPA for incident hypoxemia."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

from whole_body_jepa.waveform_jepa import WaveformEventJEPA


PATIENT_SEEDS = (7, 19, 31, 43, 59, 71, 83)
FUTURE_COLUMNS = (
    "future_spo2_min",
    "future_spo2_mean",
    "future_spo2_std",
    "future_low_fraction",
    "future_observation_count",
)


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class MatrixTransform:
    def __init__(self) -> None:
        self.median: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "MatrixTransform":
        values = np.asarray(values, dtype=float)
        self.median = np.nanmedian(values, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = np.where(np.isfinite(values), values, self.median)
        self.scale = np.nanstd(filled, axis=0)
        self.scale = np.where(self.scale >= 1e-4, self.scale, 1.0)
        return self

    def transform(self, values: np.ndarray, *, add_missing: bool = True) -> np.ndarray:
        if self.median is None or self.scale is None:
            raise RuntimeError("transform must be fitted")
        values = np.asarray(values, dtype=float)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.median, values)
        normalized = np.clip((filled - self.median) / self.scale, -12.0, 12.0)
        if add_missing:
            return np.concatenate((normalized, missing.astype(float)), axis=1).astype(np.float32)
        return normalized.astype(np.float32)


def _columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    def supported(column: str) -> bool:
        return bool(pd.to_numeric(frame[column], errors="coerce").notna().any())

    numeric = sorted(
        column for column in frame if column.startswith("num_") and supported(column)
    )
    added = sorted(
        column
        for column in frame
        if column.startswith(("ctx_", "intent_", "observed_")) and supported(column)
        and not column.endswith(("_observed", "_age_h"))
    )
    all_waveform = sorted(
        column
        for column in frame
        if (column.startswith("wf_") or column == "waveform_history_seconds")
        and supported(column)
    )
    quality = sorted(
        column
        for column in all_waveform
        if any(
            token in column
            for token in ("_available", "_lead_count", "_valid_fraction", "_flatline_fraction")
        )
        or column == "waveform_history_seconds"
    )
    waveform = []
    for column in all_waveform:
        if column in quality:
            continue
        arbitrary_amplitude = any(
            group in column for group in ("_ecg_", "_pleth_", "_respiration_")
        ) and column.endswith(("_level", "_variability", "_hypotension_fraction"))
        if not arbitrary_amplitude:
            waveform.append(column)
    if not numeric or not added or not waveform:
        raise ValueError("cohort is missing numeric, clinical-context, or waveform features")
    return numeric, added, waveform, quality


def _patient_permutation(subjects: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = np.unique(subjects.astype(str))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(unique, shuffled))
    indices_by_subject = {
        subject: np.flatnonzero(subjects.astype(str) == subject) for subject in unique
    }
    output = np.arange(len(subjects))
    for subject in unique:
        source = indices_by_subject[mapping[subject]]
        target = indices_by_subject[subject]
        output[target] = np.resize(source, len(target))
    return output


def _variant_arrays(
    context: np.ndarray,
    waveform: np.ndarray,
    subjects: np.ndarray,
    *,
    variant: str,
    numeric_width: int,
    added_width: int,
    seed: int,
    partitions: tuple[np.ndarray, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    context = context.copy()
    waveform = waveform.copy()
    if variant == "full":
        return context, waveform
    if variant == "table_baseline":
        context[:, numeric_width : numeric_width + added_width] = 0.0
        raw_width = context.shape[1] // 2
        context[:, raw_width + numeric_width : raw_width + numeric_width + added_width] = 0.0
        waveform[:] = 0.0
        return context, waveform
    if variant == "waveform_only":
        context[:, numeric_width : numeric_width + added_width] = 0.0
        raw_width = context.shape[1] // 2
        context[:, raw_width + numeric_width : raw_width + numeric_width + added_width] = 0.0
        return context, waveform
    if variant == "no_intent":
        return context, waveform
    if variant == "placebo":
        permutation = np.arange(len(subjects))
        for partition_index, partition in enumerate(partitions):
            local = np.asarray(partition, dtype=int)
            if not len(local):
                continue
            local_permutation = _patient_permutation(
                subjects[local], stable_seed(seed, partition_index)
            )
            permutation[local] = local[local_permutation]
        raw_width = context.shape[1] // 2
        added = slice(numeric_width, numeric_width + added_width)
        missing_added = slice(raw_width + numeric_width, raw_width + numeric_width + added_width)
        context[:, added] = context[permutation, added]
        context[:, missing_added] = context[permutation, missing_added]
        waveform = waveform[permutation]
        return context, waveform
    raise ValueError(f"unknown variant: {variant}")


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    grid = np.geomspace(0.35, 4.0, 80)
    losses = []
    for temperature in grid:
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -30, 30)))
        losses.append(
            -np.mean(
                labels * np.log(np.clip(probability, 1e-7, 1.0))
                + (1.0 - labels) * np.log(np.clip(1.0 - probability, 1e-7, 1.0))
            )
        )
    return float(grid[int(np.argmin(losses))])


def _train_model(
    context: np.ndarray,
    waveform: np.ndarray,
    future: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> tuple[WaveformEventJEPA, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WaveformEventJEPA(
        context.shape[1], waveform.shape[1], future.shape[1], hidden_dim=64, latent_dim=24
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    best_state = None
    best_loss = float("inf")
    stale = 0
    order_rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = fit_indices.copy()
        order_rng.shuffle(order)
        for start in range(0, len(order), 128):
            batch = order[start : start + 128]
            output = model(
                torch.from_numpy(context[batch]),
                torch.from_numpy(waveform[batch]),
                torch.from_numpy(future[batch]),
            )
            loss = model.loss(output, torch.from_numpy(labels[batch]).float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.update_target_encoder()
        model.eval()
        with torch.no_grad():
            calibration_output = model(
                torch.from_numpy(context[calibration_indices]),
                torch.from_numpy(waveform[calibration_indices]),
                torch.from_numpy(future[calibration_indices]),
            )
            validation_loss = float(
                model.loss(
                    calibration_output,
                    torch.from_numpy(labels[calibration_indices]).float(),
                )
            )
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 12:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(context[calibration_indices]),
            torch.from_numpy(waveform[calibration_indices]),
        )["logit"].numpy()
    return model, _fit_temperature(logits, labels[calibration_indices])


def _predict(
    model: WaveformEventJEPA,
    temperature: float,
    context: np.ndarray,
    waveform: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(
            torch.from_numpy(context[indices]), torch.from_numpy(waveform[indices])
        )["logit"].numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -30, 30)))


def _ece(labels: np.ndarray, probability: np.ndarray) -> float:
    output = 0.0
    for left, right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (probability >= left) & (probability <= right if right == 1 else probability < right)
        if mask.any():
            output += float(mask.mean()) * abs(float(labels[mask].mean() - probability[mask].mean()))
    return float(output)


def _paired_ci(
    labels: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    subjects: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    row_delta = (candidate - labels) ** 2 - (comparator - labels) ** 2
    unique = np.unique(subjects.astype(str))
    patient_delta = np.asarray(
        [row_delta[subjects.astype(str) == subject].mean() for subject in unique]
    )
    rng = np.random.default_rng(seed)
    draws = patient_delta[
        rng.integers(0, len(patient_delta), size=(repeats, len(patient_delta)))
    ].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "delta_brier": float(patient_delta.mean()),
        "ci95": [float(low), float(high)],
        "pass": bool(high < 0.0),
        "patients": int(len(unique)),
    }


def _evaluate(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    subjects: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    full = probabilities["full"]
    return {
        "rows": int(len(labels)),
        "patients": int(len(np.unique(subjects))),
        "prevalence": float(labels.mean()),
        "brier": {name: float(brier_score_loss(labels, probability)) for name, probability in probabilities.items()},
        "auc": {name: float(roc_auc_score(labels, probability)) if len(np.unique(labels)) > 1 else None for name, probability in probabilities.items()},
        "ece": {name: _ece(labels, probability) for name, probability in probabilities.items()},
        "full_vs_table": _paired_ci(labels, full, probabilities["table_baseline"], subjects, seed=seed, repeats=repeats),
        "full_vs_placebo": _paired_ci(labels, full, probabilities["placebo"], subjects, seed=seed + 1, repeats=repeats),
        "waveform_only_vs_table": _paired_ci(labels, probabilities["waveform_only"], probabilities["table_baseline"], subjects, seed=seed + 2, repeats=repeats),
    }


def _split_fit_calibration(subjects: np.ndarray, train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_subjects = np.unique(subjects[train].astype(str))
    calibration_subjects = {
        subject for subject in train_subjects if stable_seed("calibration", seed, subject) % 5 == 0
    }
    calibration = train & np.isin(subjects.astype(str), list(calibration_subjects))
    fit = train & ~calibration
    if len(np.unique(subjects[fit])) < 30 or len(np.unique(subjects[calibration])) < 8:
        raise ValueError("insufficient patient-disjoint fit/calibration support")
    return np.flatnonzero(fit), np.flatnonzero(calibration)


def _run_split(
    raw_context: np.ndarray,
    raw_waveform: np.ndarray,
    raw_future: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    numeric_width: int,
    added_width: int,
    seed: int,
    epochs: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_indices, calibration_indices = _split_fit_calibration(subjects, train, seed)
    context_transform = MatrixTransform().fit(raw_context[fit_indices])
    waveform_transform = MatrixTransform().fit(raw_waveform[fit_indices])
    future_transform = MatrixTransform().fit(raw_future[fit_indices])
    context = context_transform.transform(raw_context)
    waveform = waveform_transform.transform(raw_waveform)
    future = future_transform.transform(raw_future, add_missing=False)
    probabilities = {}
    states = {}
    for variant in ("table_baseline", "waveform_only", "placebo", "full"):
        variant_context, variant_waveform = _variant_arrays(
            context,
            waveform,
            subjects,
            variant=variant,
            numeric_width=numeric_width,
            added_width=added_width,
            seed=stable_seed("placebo", seed, variant),
            partitions=(fit_indices, calibration_indices, np.flatnonzero(test)),
        )
        model, temperature = _train_model(
            variant_context,
            variant_waveform,
            future,
            labels,
            fit_indices,
            calibration_indices,
            seed=stable_seed("model", seed, variant),
            epochs=epochs,
        )
        probabilities[variant] = _predict(
            model, temperature, variant_context, variant_waveform, np.flatnonzero(test)
        )
        states[variant] = {"state_dict": model.state_dict(), "temperature": temperature}
    states["preprocessing"] = {
        "context_median": context_transform.median,
        "context_scale": context_transform.scale,
        "waveform_median": waveform_transform.median,
        "waveform_scale": waveform_transform.scale,
        "future_median": future_transform.median,
        "future_scale": future_transform.scale,
        "missing_indicators_appended": True,
    }
    report = _evaluate(
        labels[test],
        probabilities,
        subjects[test],
        seed=stable_seed("ci", seed),
        repeats=repeats,
    )
    return report, states


def run(args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.read_parquet(args.cohort)
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"])
    numeric_columns, added_columns, waveform_columns, quality_columns = _columns(frame)
    context_columns = numeric_columns + added_columns
    raw_context = frame[context_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    raw_waveform = frame[waveform_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    raw_future = frame[list(FUTURE_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    labels = frame["event"].to_numpy(dtype=np.float32)
    subjects = frame["subject_id"].astype(str).to_numpy()

    patient_reports = []
    for seed in PATIENT_SEEDS:
        train = np.asarray(
            [stable_seed("outer", seed, subject) % 4 != 0 for subject in subjects], dtype=bool
        )
        report, _ = _run_split(
            raw_context,
            raw_waveform,
            raw_future,
            labels,
            subjects,
            train,
            ~train,
            numeric_width=len(numeric_columns),
            added_width=len(added_columns),
            seed=seed,
            epochs=args.epochs,
            repeats=args.bootstrap_repeats,
        )
        report["seed"] = seed
        patient_reports.append(report)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "full_vs_table": report["full_vs_table"],
                    "full_vs_placebo": report["full_vs_placebo"],
                }
            ),
            flush=True,
        )

    careunits = frame["careunit"].astype(str).to_numpy()
    unique_careunits = sorted(np.unique(careunits))
    heldout_careunits = {
        value for value in unique_careunits if stable_seed("careunit", value) % 3 == 0
    }
    careunit_test = np.isin(careunits, list(heldout_careunits))
    careunit_report, _ = _run_split(
        raw_context,
        raw_waveform,
        raw_future,
        labels,
        subjects,
        ~careunit_test,
        careunit_test,
        numeric_width=len(numeric_columns),
        added_width=len(added_columns),
        seed=9001,
        epochs=args.epochs,
        repeats=args.bootstrap_repeats,
    )
    careunit_report["heldout_careunits"] = sorted(heldout_careunits)

    order = np.argsort(frame["anchor_time"].to_numpy(dtype="datetime64[ns]"), kind="stable")
    boundary = int(round(0.70 * len(order)))
    time_test = np.zeros(len(frame), dtype=bool)
    time_test[order[boundary:]] = True
    time_report, _ = _run_split(
        raw_context,
        raw_waveform,
        raw_future,
        labels,
        subjects,
        ~time_test,
        time_test,
        numeric_width=len(numeric_columns),
        added_width=len(added_columns),
        seed=9901,
        epochs=args.epochs,
        repeats=args.bootstrap_repeats,
    )

    internal_pass = bool(
        all(report["full_vs_table"]["pass"] and report["full_vs_placebo"]["pass"] for report in patient_reports)
        and careunit_report["full_vs_table"]["pass"]
        and careunit_report["full_vs_placebo"]["pass"]
        and time_report["full_vs_table"]["pass"]
        and time_report["full_vs_placebo"]["pass"]
        and careunit_report["ece"]["full"] <= 0.05
        and time_report["ece"]["full"] <= 0.05
    )

    all_train = np.ones(len(frame), dtype=bool)
    _, final_states = _run_split(
        raw_context,
        raw_waveform,
        raw_future,
        labels,
        subjects,
        all_train,
        np.ones(len(frame), dtype=bool),
        numeric_width=len(numeric_columns),
        added_width=len(added_columns),
        seed=20260826,
        epochs=args.epochs,
        repeats=max(50, args.bootstrap_repeats // 4),
    )
    artifact = {
        "schema": "whole_body_jepa.waveform_event_jepa.v1",
        "event": args.event_name,
        "context_columns": context_columns,
        "waveform_columns": waveform_columns,
        "waveform_quality_columns": quality_columns,
        "future_columns": list(FUTURE_COLUMNS),
        "architecture": {
            "shared_context_encoder": True,
            "multiscale_waveform_encoder": True,
            "ema_target_encoder": True,
            "future_latent_prediction": True,
            "event_probability_head": True,
        },
        "full_state_dict": final_states["full"]["state_dict"],
        "temperature": final_states["full"]["temperature"],
        "preprocessing": final_states["preprocessing"],
        "internal_gate_pass": internal_pass,
        "runtime_eligible": False,
        "runtime_blocker": "independent_external_hospital_waveform_lockbox_required",
        "abstention_required_when": [
            "waveform_quality_insufficient",
            "no_recent_spo2",
            "clinical_context_stale",
            "external_event_definition_not_validated",
        ],
        "causal_claim_allowed": False,
        "clinical_decision_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    report = {
        "schema": "whole_body_jepa.waveform_event_jepa_audit.v1",
        "rows": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "event_prevalence": float(labels.mean()),
        "feature_counts": {
            "numeric_history": len(numeric_columns),
            "bedside_and_intent": len(added_columns),
            "measurement_pure_waveform": len(waveform_columns),
            "waveform_quality_for_abstention_only": len(quality_columns),
        },
        "patient_splits": patient_reports,
        "careunit_holdout": careunit_report,
        "forward_time_holdout": time_report,
        "internal_gate_pass": internal_pass,
        "runtime_eligible": False,
        "external_waveform_lockbox_complete": False,
        "boundary": {
            "future_treatment_used": False,
            "event_risk_only": True,
            "continuous_value_forecast_authority": False,
            "causal_claim_allowed": False,
            "clinical_decision_allowed": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"internal_gate_pass": internal_pass, "runtime_eligible": False}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/waveform_event_jepa_candidate.pt"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/waveform_event_jepa_audit.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    parser.add_argument("--event-name", default="incident_sustained_hypoxemia_1h")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
