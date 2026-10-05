#!/usr/bin/env python3
"""Ablate bedside assessment and observed treatment context for event risk."""

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
    _forward_time_split,
)
from scripts.train_waveform_event_jepa import (
    MatrixTransform,
    PATIENT_SEEDS,
    _ece,
    _paired_ci,
    _patient_permutation,
    _split_fit_calibration,
    stable_seed,
)


def _supported(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]


def _feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = _supported(frame, sorted(column for column in frame if column.startswith("num_")))
    waveform = _supported(
        frame,
        sorted(
            column
            for column in frame
            if column.startswith("wf_")
            and not any(
                token in column
                for token in (
                    "_available",
                    "_lead_count",
                    "_valid_fraction",
                    "_flatline_fraction",
                )
            )
            and not (
                any(group in column for group in ("_ecg_", "_pleth_", "_respiration_"))
                and column.endswith(("_level", "_variability", "_hypotension_fraction"))
            )
        ),
    )
    bedside = _supported(frame, sorted(column for column in frame if column.startswith("ctx_")))
    treatment = _supported(
        frame,
        sorted(column for column in frame if column.startswith(("intent_", "observed_", "tx_"))),
    )
    if not numeric or not waveform or not bedside or not treatment:
        raise ValueError("cohort lacks a required feature group")
    return {
        "measurement": numeric + waveform,
        "bedside": bedside,
        "treatment": treatment,
    }


def _zero_group(values: np.ndarray, indices: np.ndarray, raw_width: int) -> np.ndarray:
    output = values.copy()
    output[:, indices] = 0.0
    output[:, raw_width + indices] = 0.0
    return output


def _variant_matrices(
    transformed: np.ndarray,
    groups: dict[str, np.ndarray],
    subjects: np.ndarray,
    partitions: tuple[np.ndarray, ...],
    seed: int,
) -> dict[str, np.ndarray]:
    raw_width = transformed.shape[1] // 2
    context = np.concatenate((groups["bedside"], groups["treatment"]))
    variants = {
        "full": transformed,
        "measurement_only": _zero_group(transformed, context, raw_width),
        "no_bedside": _zero_group(transformed, groups["bedside"], raw_width),
        "no_treatment": _zero_group(transformed, groups["treatment"], raw_width),
    }
    placebo = transformed.copy()
    for partition_index, partition in enumerate(partitions):
        local = np.asarray(partition, dtype=int)
        permutation = _patient_permutation(
            subjects[local], stable_seed(seed, "context_placebo", partition_index)
        )
        source = local[permutation]
        placebo[np.ix_(local, context)] = transformed[np.ix_(source, context)]
        placebo[np.ix_(local, raw_width + context)] = transformed[
            np.ix_(source, raw_width + context)
        ]
    variants["shuffled_context_placebo"] = placebo
    return variants


def _run_split(
    raw_values: np.ndarray,
    group_indices: dict[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    seed: int,
    epochs: int,
    repeats: int,
) -> dict[str, Any]:
    fit, calibration = _split_fit_calibration(subjects, train, seed)
    test_rows = np.flatnonzero(test)
    transform = MatrixTransform().fit(raw_values[fit])
    transformed = transform.transform(raw_values)
    variants = _variant_matrices(
        transformed,
        group_indices,
        subjects,
        (fit, calibration, test_rows),
        seed,
    )
    probabilities = {}
    for name, values in variants.items():
        model, temperature = _fit_dense(
            values,
            labels,
            fit,
            calibration,
            seed=stable_seed(seed, name),
            epochs=epochs,
        )
        probabilities[name] = _dense_probability(
            model, temperature, values, test_rows
        )
    truth = labels[test_rows]
    test_subjects = subjects[test_rows]
    report = {
        "rows": int(len(test_rows)),
        "patients": int(len(np.unique(test_subjects))),
        "prevalence": float(truth.mean()),
        "brier": {
            name: float(brier_score_loss(truth, probability))
            for name, probability in probabilities.items()
        },
        "auc": {
            name: float(roc_auc_score(truth, probability))
            if len(np.unique(truth)) > 1
            else None
            for name, probability in probabilities.items()
        },
        "ece": {name: _ece(truth, probability) for name, probability in probabilities.items()},
    }
    for comparator in (
        "measurement_only",
        "no_bedside",
        "no_treatment",
        "shuffled_context_placebo",
    ):
        report[f"full_vs_{comparator}"] = _paired_ci(
            truth,
            probabilities["full"],
            probabilities[comparator],
            test_subjects,
            seed=stable_seed(seed, comparator),
            repeats=repeats,
        )
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.read_parquet(args.cohort)
    columns = _feature_groups(frame)
    ordered = columns["measurement"] + columns["bedside"] + columns["treatment"]
    offsets = {
        "measurement": np.arange(len(columns["measurement"]), dtype=int),
        "bedside": np.arange(
            len(columns["measurement"]),
            len(columns["measurement"]) + len(columns["bedside"]),
            dtype=int,
        ),
        "treatment": np.arange(
            len(columns["measurement"]) + len(columns["bedside"]),
            len(ordered),
            dtype=int,
        ),
    }
    raw_values = frame[ordered].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    labels = frame["event"].to_numpy(np.float32)
    subjects = frame["subject_id"].astype(str).to_numpy()
    patient_reports = []
    for seed in PATIENT_SEEDS:
        train = np.asarray(
            [stable_seed("context_outer", seed, subject) % 4 != 0 for subject in subjects]
        )
        report = _run_split(
            raw_values,
            offsets,
            labels,
            subjects,
            train,
            ~train,
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
                    **{
                        key: value
                        for key, value in report.items()
                        if key.startswith("full_vs_")
                    },
                }
            ),
            flush=True,
        )

    careunits = frame["careunit"].astype(str).to_numpy()
    care_train, care_test, heldout = _careunit_split(subjects, careunits)
    care_report = _run_split(
        raw_values,
        offsets,
        labels,
        subjects,
        care_train,
        care_test,
        seed=9001,
        epochs=args.epochs,
        repeats=args.bootstrap_repeats,
    )
    care_report["heldout_careunits"] = heldout
    time_train, time_test = _forward_time_split(subjects, frame["anchor_time"])
    time_report = _run_split(
        raw_values,
        offsets,
        labels,
        subjects,
        time_train,
        time_test,
        seed=9901,
        epochs=args.epochs,
        repeats=args.bootstrap_repeats,
    )
    comparisons = (
        "full_vs_measurement_only",
        "full_vs_no_bedside",
        "full_vs_no_treatment",
        "full_vs_shuffled_context_placebo",
    )
    internal_pass = bool(
        all(all(item[key]["pass"] for key in comparisons) for item in patient_reports)
        and all(
            all(item[key]["pass"] for key in comparisons)
            and item["ece"]["full"] <= 0.05
            for item in (care_report, time_report)
        )
    )
    result = {
        "schema": "whole_body_jepa.waveform_clinical_context_ablation.v1",
        "event": args.event_name,
        "rows": int(len(frame)),
        "patients": int(frame["subject_id"].nunique()),
        "feature_columns": columns,
        "patient_splits": patient_reports,
        "careunit_holdout": care_report,
        "forward_time_holdout": time_report,
        "internal_gate_pass": internal_pass,
        "runtime_eligible": False,
        "external_waveform_lockbox_complete": False,
        "causal_claim_allowed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"internal_gate_pass": internal_pass}, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
