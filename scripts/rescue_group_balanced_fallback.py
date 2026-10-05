#!/usr/bin/env python3
"""Test hospital-balanced residual heads for unresolved 6h release cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from scripts.validate_subtype_external import _load_cell, stable_seed
from whole_body_jepa.runtime import load_release


TARGETS = ("creatinine", "hematocrit", "hemoglobin")
PATIENT_SEEDS = (7, 19, 31, 43, 59, 71, 83)


class RobustRidge:
    def __init__(self, alpha: float, power: float) -> None:
        self.alpha = float(alpha)
        self.power = float(power)
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.model: Ridge | None = None

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        groups: np.ndarray,
    ) -> "RobustRidge":
        matrix = np.asarray(features, dtype=np.float64)
        self.center = np.median(matrix, axis=0)
        lower, upper = np.quantile(matrix, [0.25, 0.75], axis=0)
        self.scale = np.where(upper - lower > 1e-6, upper - lower, 1.0)
        standardized = np.clip((matrix - self.center) / self.scale, -12.0, 12.0)
        labels, counts = np.unique(groups.astype(str), return_counts=True)
        count_by_group = dict(zip(labels, counts))
        weights = np.asarray(
            [count_by_group[str(group)] ** (-self.power) for group in groups],
            dtype=np.float64,
        )
        weights /= weights.mean()
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(standardized, target, sample_weight=weights)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None or self.center is None or self.scale is None:
            raise RuntimeError("model is not fitted")
        matrix = np.asarray(features, dtype=np.float64)
        standardized = np.clip((matrix - self.center) / self.scale, -12.0, 12.0)
        return self.model.predict(standardized)


def _domain_folds(groups: np.ndarray, seed: int) -> list[np.ndarray]:
    unique = np.unique(groups.astype(str))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return [np.asarray(part, dtype=object) for part in np.array_split(unique, min(5, len(unique)))]


def _patient_means(values: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    keys = subjects.astype(str)
    return np.asarray([values[keys == key].mean() for key in np.unique(keys)])


def _point_delta(
    truth: np.ndarray,
    candidate: np.ndarray,
    persistence: np.ndarray,
    groups: np.ndarray,
) -> float:
    loss_delta = np.abs(candidate - truth) - np.abs(persistence - truth)
    return float(_patient_means(loss_delta, groups).mean())


def _select(payload: dict[str, Any], scale: float, target: str) -> dict[str, Any] | None:
    features = payload["features"]
    truth = payload["truth"]
    current = payload["frame"][f"{target}_t"].to_numpy(dtype=np.float64)
    normalized_delta = (truth - current) / scale
    hospitals = payload["hospitals"].astype(str)
    subjects = payload["subjects"].astype(str)
    time = np.asarray(payload["time"]).astype("datetime64[ns]")
    consistency = np.full(len(truth), 0.0, dtype=np.float64)
    consistency_limits = (1.0e9,)
    if target in {"hematocrit", "hemoglobin"}:
        hematocrit = payload["frame"]["hematocrit_t"].to_numpy(dtype=np.float64)
        hemoglobin = payload["frame"]["hemoglobin_t"].to_numpy(dtype=np.float64)
        consistency = np.abs(hematocrit - 3.0 * hemoglobin)
        consistency = np.where(np.isfinite(consistency), consistency, np.inf)
        consistency_limits = (0.5, 1.0, 2.0, 4.0, 1.0e9)
    folds = _domain_folds(hospitals, stable_seed("group_balanced_folds", target))
    candidates = []
    for alpha in (10.0, 100.0, 1000.0):
        for power in (0.5, 1.0):
            raw = np.full(len(truth), np.nan, dtype=np.float64)
            for heldout in folds:
                test = np.isin(hospitals, heldout)
                train = ~test
                model = RobustRidge(alpha, power).fit(
                    features[train], normalized_delta[train], hospitals[train]
                )
                raw[test] = model.predict(features[test]) * scale
            if not np.isfinite(raw).all():
                continue
            for quantile in (0.0, 0.25, 0.5, 0.75):
                threshold = float(np.quantile(np.abs(raw), quantile))
                for consistency_limit in consistency_limits:
                    active = (np.abs(raw) >= threshold) & (
                        consistency <= consistency_limit
                    )
                    for shrinkage in (0.25, 0.5, 0.75, 1.0):
                        prediction = current + shrinkage * raw * active
                        patient_delta = _point_delta(
                            truth, prediction, current, subjects
                        )
                        hospital_delta = _point_delta(
                            truth, prediction, current, hospitals
                        )
                        late = np.argsort(time, kind="stable")[int(0.70 * len(time)) :]
                        time_delta = _point_delta(
                            truth[late], prediction[late], current[late], subjects[late]
                        )
                        robust = max(patient_delta, hospital_delta, time_delta)
                        candidates.append(
                            {
                                "alpha": alpha,
                                "hospital_weight_power": power,
                                "abstention_quantile": quantile,
                                "threshold": threshold,
                                "consistency_limit": consistency_limit,
                                "shrinkage": shrinkage,
                                "patient_delta": patient_delta,
                                "hospital_delta": hospital_delta,
                                "forward_time_delta": time_delta,
                                "robust_objective": robust,
                                "active_fraction": float(active.mean()),
                            }
                        )
    passed = [item for item in candidates if item["robust_objective"] < 0.0]
    return min(passed, key=lambda item: item["robust_objective"]) if passed else None


def _paired_ci(
    truth: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    delta = np.abs(candidate - truth) - np.abs(comparator - truth)
    means = _patient_means(delta, groups)
    rng = np.random.default_rng(seed)
    draws = means[
        rng.integers(0, len(means), size=(repeats, len(means)))
    ].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "groups": int(len(means)),
        "delta_mae": float(means.mean()),
        "ci95": [float(low), float(high)],
        "pass": bool(high < 0.0),
    }


def _evaluate_view(
    truth: np.ndarray,
    candidate: np.ndarray,
    persistence: np.ndarray,
    placebo: np.ndarray,
    subjects: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    versus_persistence = _paired_ci(
        truth, candidate, persistence, subjects, seed=seed, repeats=repeats
    )
    versus_placebo = _paired_ci(
        truth, candidate, placebo, subjects, seed=seed + 1, repeats=repeats
    )
    return {
        "rows": int(len(truth)),
        "patients": int(len(np.unique(subjects))),
        "candidate_mae": float(np.mean(np.abs(candidate - truth))),
        "persistence_mae": float(np.mean(np.abs(persistence - truth))),
        "placebo_mae": float(np.mean(np.abs(placebo - truth))),
        "versus_persistence": versus_persistence,
        "versus_placebo": versus_placebo,
        "pass": bool(versus_persistence["pass"] and versus_placebo["pass"]),
    }


def _external_audit(
    discovery: dict[str, Any],
    lockbox: dict[str, Any],
    selection: dict[str, Any],
    scale: float,
    target: str,
    repeats: int,
) -> tuple[dict[str, Any], RobustRidge]:
    current = lockbox["frame"][f"{target}_t"].to_numpy(dtype=np.float64)
    discovery_current = discovery["frame"][f"{target}_t"].to_numpy(dtype=np.float64)
    discovery_delta = (discovery["truth"] - discovery_current) / scale
    model = RobustRidge(
        selection["alpha"], selection["hospital_weight_power"]
    ).fit(discovery["features"], discovery_delta, discovery["hospitals"])
    raw = model.predict(lockbox["features"]) * scale
    consistency = np.full(len(raw), 0.0, dtype=np.float64)
    if target in {"hematocrit", "hemoglobin"}:
        hematocrit = lockbox["frame"]["hematocrit_t"].to_numpy(dtype=np.float64)
        hemoglobin = lockbox["frame"]["hemoglobin_t"].to_numpy(dtype=np.float64)
        consistency = np.abs(hematocrit - 3.0 * hemoglobin)
        consistency = np.where(np.isfinite(consistency), consistency, np.inf)
    active = (np.abs(raw) >= selection["threshold"]) & (
        consistency <= selection["consistency_limit"]
    )
    candidate = current + selection["shrinkage"] * raw * active

    rng = np.random.default_rng(stable_seed("group_balanced_placebo", target))
    shuffled = discovery_delta.copy()
    rng.shuffle(shuffled)
    placebo_model = RobustRidge(
        selection["alpha"], selection["hospital_weight_power"]
    ).fit(discovery["features"], shuffled, discovery["hospitals"])
    placebo_raw = placebo_model.predict(lockbox["features"]) * scale
    placebo = current + selection["shrinkage"] * placebo_raw * (
        (np.abs(placebo_raw) >= selection["threshold"])
        & (consistency <= selection["consistency_limit"])
    )

    patient_splits = []
    for split_seed in PATIENT_SEEDS:
        mask = np.asarray(
            [stable_seed("fallback_patient", split_seed, value) % 4 == 0 for value in lockbox["subjects"]]
        )
        patient_splits.append(
            {
                "seed": split_seed,
                **_evaluate_view(
                    lockbox["truth"][mask],
                    candidate[mask],
                    current[mask],
                    placebo[mask],
                    lockbox["subjects"][mask],
                    seed=stable_seed("fallback_patient_ci", target, split_seed),
                    repeats=repeats,
                ),
            }
        )
    careunit = _evaluate_view(
        lockbox["truth"],
        candidate,
        current,
        placebo,
        lockbox["careunits"],
        seed=stable_seed("fallback_careunit", target),
        repeats=repeats,
    )
    order = np.argsort(np.asarray(lockbox["time"]).astype("datetime64[ns]"), kind="stable")
    late = order[int(0.70 * len(order)) :]
    forward = _evaluate_view(
        lockbox["truth"][late],
        candidate[late],
        current[late],
        placebo[late],
        lockbox["subjects"][late],
        seed=stable_seed("fallback_time", target),
        repeats=repeats,
    )
    passed = all(item["pass"] for item in patient_splits) and careunit["pass"] and forward["pass"]
    return (
        {
            "runtime_eligible": bool(passed),
            "active_fraction": float(active.mean()),
            "patient_splits": patient_splits,
            "careunit_holdout": careunit,
            "forward_time": forward,
        },
        model,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    assets = load_release(args.model_dir)
    registry = {cell["cell"]: cell for cell in assets.registry["cells"]}
    reports = {}
    for target in args.targets:
        cell = f"{target}@6h"
        print(f"auditing {cell}", flush=True)
        discovery = _load_cell(
            database="eICU",
            project_root=args.project_root,
            eicu_root=args.eicu_root,
            target=target,
            horizon=6,
            cell_contract=registry[cell],
            assets=assets,
            max_rows=args.max_rows,
            max_rows_per_subject=args.max_rows_per_subject,
            batch_size=args.batch_size,
            include_complete_history=True,
        )
        lockbox = _load_cell(
            database="MIMIC",
            project_root=args.project_root,
            eicu_root=args.eicu_root,
            target=target,
            horizon=6,
            cell_contract=registry[cell],
            assets=assets,
            max_rows=args.max_rows,
            max_rows_per_subject=args.max_rows_per_subject,
            batch_size=args.batch_size,
        )
        if discovery is None or lockbox is None:
            reports[cell] = {"runtime_eligible": False, "reason": "missing_support"}
            continue
        scale = float(assets.scaler.scales[list(assets.checkpoint["variables"]).index(target)])
        selection = _select(discovery, scale, target)
        if selection is None:
            reports[cell] = {
                "runtime_eligible": False,
                "reason": "no_eicu_hospital_robust_candidate",
            }
            continue
        external, _ = _external_audit(
            discovery, lockbox, selection, scale, target, args.bootstrap_repeats
        )
        reports[cell] = {
            "selection_database": "eICU",
            "external_database": "MIMIC-IV",
            "selection": selection,
            "external": external,
            "runtime_eligible": external["runtime_eligible"],
        }
    output = {
        "schema": "whole_body_jepa.group_balanced_fallback_rescue.v1",
        "method": "hospital-balanced ridge residual with discovery-only shrinkage and abstention",
        "reports": reports,
        "validated_cells": sorted(
            cell for cell, report in reports.items() if report["runtime_eligible"]
        ),
        "fresh_unseen_lockbox_required_for_formal_promotion": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"validated_cells": output["validated_cells"]}, indent=2), flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--eicu-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/group_balanced_fallback_rescue.json")
    )
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--max-rows-per-subject", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
