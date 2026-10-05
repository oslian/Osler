#!/usr/bin/env python3
"""Train and audit calibrated factual hypoxemia event-risk heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.validate_subtype_external import _load_cell, stable_seed
from whole_body_jepa.runtime import load_release


EVENT_THRESHOLD = 92.0
PATIENT_SEEDS = (7, 19, 31, 43, 59, 71, 83)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_calibrated_head(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    *,
    seed: int,
) -> CalibratedClassifierCV:
    groups = np.asarray(subjects).astype(str)
    unique = np.unique(groups)
    calibration_subjects = {
        subject
        for subject in unique
        if stable_seed("event_calibration", seed, subject) % 5 == 0
    }
    calibration = np.isin(groups, list(calibration_subjects))
    fit = ~calibration
    if min(int(fit.sum()), int(calibration.sum())) < 100:
        raise ValueError("event head requires patient-disjoint fit/calibration data")
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.20, max_iter=500, solver="lbfgs"),
    )
    base.fit(features[fit], labels[fit])
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(base), method="sigmoid"
    )
    calibrated.fit(features[calibration], labels[calibration])
    return calibrated


def _ece(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            total += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probability[mask].mean())
            )
    return float(total)


def _paired_patient_ci(
    labels: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    subjects: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    delta = (candidate - labels) ** 2 - (comparator - labels) ** 2
    groups = np.asarray(subjects).astype(str)
    patient_delta = np.asarray(
        [delta[groups == group].mean() for group in np.unique(groups)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(patient_delta), size=(repeats, len(patient_delta))
    )
    draws = patient_delta[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "patient_equal_delta_brier": float(patient_delta.mean()),
        "ci95": [float(low), float(high)],
        "pass": bool(high < 0.0),
        "patients": int(len(patient_delta)),
    }


def _evaluate(
    labels: np.ndarray,
    probability: np.ndarray,
    current_event: np.ndarray,
    subjects: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    prevalence = np.full(len(labels), float(labels.mean()), dtype=np.float64)
    versus_persistence = _paired_patient_ci(
        labels,
        probability,
        current_event,
        subjects,
        seed=seed,
        repeats=repeats,
    )
    versus_prevalence = _paired_patient_ci(
        labels,
        probability,
        prevalence,
        subjects,
        seed=seed + 1,
        repeats=repeats,
    )
    ece = _ece(labels, probability)
    return {
        "rows": int(len(labels)),
        "patients": int(len(np.unique(subjects))),
        "prevalence": float(labels.mean()),
        "candidate_brier": float(np.mean((probability - labels) ** 2)),
        "persistence_brier": float(np.mean((current_event - labels) ** 2)),
        "prevalence_brier": float(np.mean((prevalence - labels) ** 2)),
        "expected_calibration_error": ece,
        "versus_persistence": versus_persistence,
        "versus_prevalence": versus_prevalence,
        "pass": bool(
            versus_persistence["pass"]
            and versus_prevalence["pass"]
            and ece <= 0.03
        ),
    }


def _patient_audit(payload: dict[str, Any], labels: np.ndarray, horizon: int, repeats: int):
    output = []
    subjects = payload["subjects"]
    current_event = (payload["frame"]["o2sat_t"].to_numpy(dtype=float) <= EVENT_THRESHOLD).astype(float)
    for seed in PATIENT_SEEDS:
        unique = np.unique(subjects)
        train_subjects = {
            subject
            for subject in unique
            if stable_seed("event_outer", horizon, seed, subject) % 4 != 0
        }
        train = np.isin(subjects, list(train_subjects))
        test = ~train
        model = _fit_calibrated_head(
            payload["features"][train], labels[train], subjects[train], seed=seed
        )
        probability = model.predict_proba(payload["features"][test])[:, 1]
        output.append(
            {
                "seed": seed,
                **_evaluate(
                    labels[test],
                    probability,
                    current_event[test],
                    subjects[test],
                    seed=stable_seed("event_patient_ci", horizon, seed),
                    repeats=repeats,
                ),
            }
        )
    return output


def _domain_audit(
    payload: dict[str, Any],
    labels: np.ndarray,
    horizon: int,
    repeats: int,
    *,
    axis: str,
):
    domain_values = payload[axis]
    probability = np.full(len(labels), np.nan, dtype=np.float64)
    current_event = (payload["frame"]["o2sat_t"].to_numpy(dtype=float) <= EVENT_THRESHOLD).astype(float)
    used = np.zeros(len(labels), dtype=bool)
    folds = []
    domains = np.unique(domain_values.astype(str))
    rng = np.random.default_rng(stable_seed("event_domain_groups", horizon, axis))
    rng.shuffle(domains)
    domain_folds = np.array_split(domains, min(5, len(domains)))
    for fold_index, heldout_domains in enumerate(domain_folds):
        test = np.isin(domain_values.astype(str), heldout_domains)
        train = ~test
        if int(test.sum()) < 100 or len(np.unique(payload["subjects"][train])) < 100:
            continue
        model = _fit_calibrated_head(
            payload["features"][train],
            labels[train],
            payload["subjects"][train],
            seed=stable_seed("event_domain", horizon, axis, fold_index),
        )
        probability[test] = model.predict_proba(payload["features"][test])[:, 1]
        used[test] = True
        folds.append(
            {
                "fold": int(fold_index),
                "domains": [str(value) for value in heldout_domains],
                "rows": int(test.sum()),
            }
        )
    if not used.any():
        return {"pass": False, "reason": "insufficient_grouped_domain_support", "folds": []}
    result = _evaluate(
        labels[used],
        probability[used],
        current_event[used],
        payload["subjects"][used],
        seed=stable_seed("event_domain_ci", horizon, axis),
        repeats=repeats,
    )
    result["folds"] = folds
    return result


def _time_audit(payload: dict[str, Any], labels: np.ndarray, horizon: int, repeats: int):
    order = np.argsort(np.asarray(payload["time"]).astype("datetime64[ns]"), kind="stable")
    boundary = int(round(0.70 * len(order)))
    train = order[:boundary]
    test = order[boundary:]
    model = _fit_calibrated_head(
        payload["features"][train],
        labels[train],
        payload["subjects"][train],
        seed=stable_seed("event_time", horizon),
    )
    probability = model.predict_proba(payload["features"][test])[:, 1]
    current_event = (payload["frame"]["o2sat_t"].to_numpy(dtype=float) <= EVENT_THRESHOLD).astype(float)
    return _evaluate(
        labels[test],
        probability,
        current_event[test],
        payload["subjects"][test],
        seed=stable_seed("event_time_ci", horizon),
        repeats=repeats,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_release(args.model_dir, require_event_heads=False)
    contracts = {
        int(cell["horizon_hours"]): cell
        for cell in assets.registry["cells"]
        if cell["target"] == "hypoxemia_event"
    }
    heads = {}
    reports = {}
    for horizon in sorted(contracts):
        cell = f"hypoxemia_event@{horizon}h"
        print(f"training {cell}", flush=True)
        payload = _load_cell(
            database="eICU",
            project_root=args.project_root,
            eicu_root=args.eicu_root,
            target="o2sat",
            horizon=horizon,
            cell_contract=contracts[horizon],
            assets=assets,
            max_rows=args.max_rows,
            max_rows_per_subject=args.max_rows_per_subject,
            batch_size=args.batch_size,
            include_complete_history=True,
        )
        if payload is None:
            raise ValueError(f"missing eICU support for {cell}")
        labels = (payload["truth"] <= EVENT_THRESHOLD).astype(np.int8)
        patient = _patient_audit(payload, labels, horizon, args.bootstrap_repeats)
        hospital = _domain_audit(
            payload,
            labels,
            horizon,
            args.bootstrap_repeats,
            axis="hospitals",
        )
        careunit = _domain_audit(
            payload,
            labels,
            horizon,
            args.bootstrap_repeats,
            axis="careunits",
        )
        forward_time = _time_audit(payload, labels, horizon, args.bootstrap_repeats)
        passed = (
            all(item["pass"] for item in patient)
            and hospital["pass"]
            and careunit["pass"]
            and forward_time["pass"]
        )
        reports[cell] = {
            "rows": int(len(labels)),
            "patients": int(len(np.unique(payload["subjects"]))),
            "patient_splits": patient,
            "hospital_holdout": hospital,
            "careunit_holdout": careunit,
            "forward_time": forward_time,
            "runtime_eligible": passed,
        }
        if passed:
            heads[cell] = _fit_calibrated_head(
                payload["features"],
                labels,
                payload["subjects"],
                seed=stable_seed("event_final", horizon),
            )

    bindings = {
        name: _sha256(args.model_dir / name)
        for name in (
            "whole_body_joint_jepa.pt",
            "unified_shared_jepa_head_bundle.joblib",
            "runtime_registry.json",
        )
    }
    artifact = {
        "schema": "whole_body_jepa.event_risk_heads.v1",
        "event_definition": "future_o2sat <= 92",
        "feature_width": 311,
        "calibration_database": "eICU-CRD",
        "external_method_evidence": "mimiciv_hypoxemia_event_external_audit_20260816.json",
        "mimic_raw_event_cohort_available_in_release_workspace": False,
        "release_binding": bindings,
        "validated_heads": sorted(heads),
        "heads": heads,
        "reports": reports,
        "causal_claim_allowed": False,
        "clinical_decision_allowed": False,
    }
    joblib.dump(artifact, args.output)
    report = {
        key: value for key, value in artifact.items() if key not in {"heads"}
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"validated_heads": artifact["validated_heads"]}, indent=2), flush=True)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--eicu-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("models/event_risk_heads.joblib"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/event_risk_head_audit.json"))
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--max-rows-per-subject", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
