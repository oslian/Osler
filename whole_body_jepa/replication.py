"""Frozen cross-database replication metrics for new factual lockboxes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "patient_id",
    "cell",
    "truth",
    "candidate",
    "persistence",
}


@dataclass(frozen=True)
class GateResult:
    groups: int
    delta: float
    ci95: tuple[float, float]
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": self.groups,
            "delta_candidate_minus_persistence": self.delta,
            "ci95": list(self.ci95),
            "pass": self.passed,
        }


def _bootstrap_group_delta(
    errors: pd.DataFrame,
    group_column: str,
    *,
    repetitions: int,
    seed: int,
) -> GateResult | None:
    grouped = (
        errors.groupby(group_column, sort=False, observed=True)[
            ["candidate_error", "persistence_error"]
        ]
        .mean()
        .dropna()
    )
    if len(grouped) < 2:
        return None
    delta = (
        grouped["candidate_error"] - grouped["persistence_error"]
    ).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    max_indices_per_batch = 2_000_000
    batch_size = max(1, min(repetitions, max_indices_per_batch // len(delta)))
    sample_batches = []
    for start in range(0, repetitions, batch_size):
        count = min(batch_size, repetitions - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        sample_batches.append(delta[indices].mean(axis=1))
    samples = np.concatenate(sample_batches)
    low, high = np.quantile(samples, [0.025, 0.975])
    return GateResult(
        groups=int(len(delta)),
        delta=float(delta.mean()),
        ci95=(float(low), float(high)),
        passed=bool(high < 0.0),
    )


def _direction_metrics(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    if "current" not in frame:
        return None, None
    current = pd.to_numeric(frame["current"], errors="coerce").to_numpy(dtype=float)
    truth = frame["truth"].to_numpy(dtype=float)
    candidate = frame["candidate"].to_numpy(dtype=float)
    observed = np.isfinite(current) & np.isfinite(truth) & np.isfinite(candidate)
    if not observed.any():
        return None, None
    true_delta = truth[observed] - current[observed]
    predicted_delta = candidate[observed] - current[observed]
    changed = np.abs(true_delta) > 1e-12
    accuracy = (
        float(np.mean(np.sign(true_delta[changed]) == np.sign(predicted_delta[changed])))
        if changed.any()
        else None
    )
    correlation = None
    if changed.sum() >= 3 and np.std(true_delta[changed]) > 0 and np.std(predicted_delta[changed]) > 0:
        correlation = float(np.corrcoef(true_delta[changed], predicted_delta[changed])[0, 1])
    return accuracy, correlation


def _coverage(frame: pd.DataFrame, group_column: str | None) -> dict[str, Any] | None:
    if not {"lower", "upper"}.issubset(frame.columns):
        return None
    lower = pd.to_numeric(frame["lower"], errors="coerce")
    upper = pd.to_numeric(frame["upper"], errors="coerce")
    valid = lower.notna() & upper.notna()
    if not valid.any():
        return None
    covered = ((frame.loc[valid, "truth"] >= lower[valid]) & (frame.loc[valid, "truth"] <= upper[valid])).astype(float)
    result: dict[str, Any] = {
        "rows": int(valid.sum()),
        "row_coverage": float(covered.mean()),
    }
    if group_column and group_column in frame:
        grouped = covered.groupby(frame.loc[valid, group_column], observed=True).mean()
        result["group_equal_coverage"] = float(grouped.mean())
        result["groups"] = int(len(grouped))
    check = result.get("group_equal_coverage", result["row_coverage"])
    result["pass_87_93"] = bool(0.87 <= check <= 0.93)
    return result


def validate_scored_lockbox(
    rows: pd.DataFrame,
    *,
    expected_cells: Iterable[str] | None = None,
    domain_column: str = "domain_id",
    time_column: str = "time_block",
    bootstrap_repetitions: int = 2000,
    seed: int = 20260826,
) -> dict[str, Any]:
    """Evaluate frozen predictions without selecting or fitting on the lockbox.

    Rows must already contain predictions made by the frozen eICU model. The
    function never fits a model, tunes a threshold, or alters a prediction.
    """

    missing = REQUIRED_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError(f"scored lockbox is missing columns: {sorted(missing)}")
    frame = rows.copy()
    for column in ("truth", "candidate", "persistence"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["patient_id", "cell", "truth", "candidate", "persistence"])
    if frame.empty:
        raise ValueError("scored lockbox has no finite evaluation rows")

    expected = set(expected_cells or ())
    reports: list[dict[str, Any]] = []
    for index, (cell, cell_frame) in enumerate(frame.groupby("cell", sort=True, observed=True)):
        cell_frame = cell_frame.copy()
        task_type = (
            str(cell_frame["task_type"].iloc[0])
            if "task_type" in cell_frame
            else "continuous_value"
        )
        if task_type == "binary_event_probability":
            cell_frame["candidate_error"] = (cell_frame["candidate"] - cell_frame["truth"]) ** 2
            cell_frame["persistence_error"] = (cell_frame["persistence"] - cell_frame["truth"]) ** 2
        else:
            cell_frame["candidate_error"] = (cell_frame["candidate"] - cell_frame["truth"]).abs()
            cell_frame["persistence_error"] = (cell_frame["persistence"] - cell_frame["truth"]).abs()

        patient = _bootstrap_group_delta(
            cell_frame,
            "patient_id",
            repetitions=bootstrap_repetitions,
            seed=seed + index,
        )
        domain = (
            _bootstrap_group_delta(
                cell_frame,
                domain_column,
                repetitions=bootstrap_repetitions,
                seed=seed + 1000 + index,
            )
            if domain_column in cell_frame
            else None
        )
        forward_time = (
            _bootstrap_group_delta(
                cell_frame,
                time_column,
                repetitions=bootstrap_repetitions,
                seed=seed + 2000 + index,
            )
            if time_column in cell_frame
            else None
        )
        direction, correlation = _direction_metrics(cell_frame)
        strict_pass = bool(
            patient
            and domain
            and forward_time
            and patient.passed
            and domain.passed
            and forward_time.passed
        )
        reports.append(
            {
                "cell": str(cell),
                "task_type": task_type,
                "rows": int(len(cell_frame)),
                "patients": int(cell_frame["patient_id"].nunique()),
                "candidate_error": float(cell_frame["candidate_error"].mean()),
                "persistence_error": float(cell_frame["persistence_error"].mean()),
                "patient_gate": patient.as_dict() if patient else None,
                "domain_gate": domain.as_dict() if domain else None,
                "forward_time_gate": forward_time.as_dict() if forward_time else None,
                "direction_accuracy": direction,
                "delta_correlation": correlation,
                "interval": _coverage(cell_frame, domain_column),
                "strict_frozen_replication_pass": strict_pass,
                "failure_reasons": [
                    name
                    for name, result in (
                        ("patient_gate", patient),
                        ("domain_gate", domain),
                        ("forward_time_gate", forward_time),
                    )
                    if result is None or not result.passed
                ],
            }
        )

    evaluated = {report["cell"] for report in reports}
    return {
        "schema": "whole_body_jepa.frozen_lockbox_validation.v1",
        "policy": "frozen_eicu_weights_no_lockbox_fitting",
        "evaluated_cells": len(evaluated),
        "strict_pass_cells": [
            report["cell"] for report in reports if report["strict_frozen_replication_pass"]
        ],
        "missing_expected_cells": sorted(expected - evaluated),
        "cells": reports,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }
