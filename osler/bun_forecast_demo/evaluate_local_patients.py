"""Evaluate the deployed BUN@3h artifact on local patient transitions.

The evaluator uses the exact serialized artifact loaded by the website.  It
requires six causal history rows from one stay, never uses future treatment or
future measurements as inputs, and reports aggregate-only factual metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from aki_renal_belief import renal_belief_state_v2_features
from bun_forecast_demo.runtime import HORIZON_HOURS, artifact


HISTORY_STEPS = 6
TARGET = "bun"
FUTURE_COLUMN = f"{TARGET}_tp{HORIZON_HOURS}"


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int = 20260814,
    samples: int = 1000,
) -> list[float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(clean, size=len(clean), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _patient_equalized(
    subject_ids: np.ndarray,
    values: np.ndarray,
) -> tuple[float, np.ndarray]:
    frame = pd.DataFrame({
        "subject_id": np.asarray(subject_ids).astype(str),
        "value": np.asarray(values, dtype=np.float64),
    })
    by_patient = frame.groupby("subject_id", sort=False)["value"].mean().to_numpy()
    return float(by_patient.mean()), by_patient


def _prepare_frame(cohort: Path) -> pd.DataFrame:
    frame = pd.read_parquet(cohort).copy()
    required = {"subject_id", "stay_id", "t", "bun_t", FUTURE_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cohort is missing required columns: {missing}")
    frame["t"] = pd.to_datetime(frame["t"])
    frame = frame.sort_values(["stay_id", "t"], kind="stable").reset_index(drop=True)
    first_time = frame.groupby("stay_id", sort=False)["t"].transform("min")
    frame["hours_since_onset"] = (
        (frame["t"] - first_time).dt.total_seconds() / 3600.0
    )
    return frame


def _history_indices(frame: pd.DataFrame, steps: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.full((len(frame), steps), -1, dtype=np.int64)
    complete = np.zeros(len(frame), dtype=bool)
    for _, rows in frame.groupby("stay_id", sort=False):
        ordered = rows.index.to_numpy(dtype=np.int64)
        if len(ordered) < steps:
            continue
        for position in range(steps - 1, len(ordered)):
            row_index = ordered[position]
            indices[row_index] = ordered[position - steps + 1 : position + 1]
            complete[row_index] = True
    return indices, complete


def _predict_batches(
    frame: pd.DataFrame,
    eligible: np.ndarray,
    history_indices: np.ndarray,
    beliefs: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loaded = artifact()
    columns = list(loaded.metadata["feature_columns"])
    physical = frame.reindex(columns=columns).apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    current = physical[eligible]
    history_rows = history_indices[eligible]
    history = physical[history_rows]
    anchor_times = frame.loc[eligible, "t"].to_numpy()
    event_times = frame.loc[history_rows.reshape(-1), "t"].to_numpy().reshape(
        history_rows.shape
    )
    elapsed = ((event_times - anchor_times[:, None]) / np.timedelta64(1, "h")).astype(
        np.float32
    )

    prep = loaded.preprocessing
    normalized_current = np.nan_to_num(
        (current - prep["current_center"]) / prep["current_scale"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    normalized_history = np.nan_to_num(
        (history - prep["current_center"]) / prep["current_scale"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    normalized_history = np.concatenate(
        (
            normalized_history,
            np.clip(elapsed[..., None] / 48.0, -1.0, 0.0),
        ),
        axis=-1,
    )
    normalized_belief = np.nan_to_num(
        (beliefs[eligible] - prep["belief_center"]) / prep["belief_scale"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    population = loaded.population_anchor.predict(current).astype(np.float32)
    target_center = float(prep["target_center"][0])
    target_scale = float(prep["target_scale"][0])
    anchors = ((population - target_center) / target_scale).astype(np.float32)
    current_target = frame.loc[eligible, "bun_t"].to_numpy(dtype=np.float32)
    normalized_target = ((current_target - target_center) / target_scale).astype(
        np.float32
    )

    points: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    loaded.model.eval()
    with torch.no_grad():
        for start in range(0, len(eligible), batch_size):
            stop = min(start + batch_size, len(eligible))
            point, scale = loaded.model(
                torch.from_numpy(normalized_current[start:stop]),
                torch.from_numpy(normalized_history[start:stop]),
                torch.ones((stop - start, HISTORY_STEPS), dtype=torch.bool),
                torch.from_numpy(normalized_belief[start:stop]),
                torch.from_numpy(anchors[start:stop]),
                torch.from_numpy(normalized_target[start:stop]),
            )
            points.append(point.numpy())
            scales.append(scale.numpy())
    personalized = np.concatenate(points) * target_scale + target_center
    physical_scale = np.concatenate(scales) * target_scale

    alpha_grid = prep["conformal_alpha"].astype(np.float64)
    quantile_grid = prep["conformal_quantile"].astype(np.float64)
    quantile = float(quantile_grid[int(np.argmin(np.abs(alpha_grid - 0.10)))])
    half_width = np.maximum(0.0, physical_scale * quantile)
    return {
        "population": population.astype(np.float64),
        "personalized": personalized.astype(np.float64),
        "half_width": half_width.astype(np.float64),
        "elapsed_min": elapsed.min(axis=1).astype(np.float64),
        "elapsed_max": elapsed.max(axis=1).astype(np.float64),
    }


def evaluate(
    cohort: Path,
    *,
    batch_size: int = 4096,
    bootstrap_samples: int = 1000,
    target_tolerance_hours: float | None = None,
) -> dict[str, Any]:
    frame = _prepare_frame(cohort)
    history_indices, complete_history = _history_indices(frame, HISTORY_STEPS)
    pair = (
        pd.to_numeric(frame["bun_t"], errors="coerce").notna()
        & pd.to_numeric(frame[FUTURE_COLUMN], errors="coerce").notna()
    ).to_numpy()
    eligible = np.flatnonzero(pair & complete_history)
    if len(eligible) == 0:
        raise ValueError("cohort has no BUN pairs with six causal history rows")

    relevant_stays = set(frame.loc[eligible, "stay_id"].tolist())
    belief_frame = frame[frame["stay_id"].isin(relevant_stays)].copy()
    beliefs_relevant = renal_belief_state_v2_features(belief_frame).to_numpy(
        dtype=np.float32
    )
    beliefs = np.full((len(frame), beliefs_relevant.shape[1]), np.nan, dtype=np.float32)
    beliefs[belief_frame.index.to_numpy(dtype=np.int64)] = beliefs_relevant

    output = _predict_batches(
        frame,
        eligible,
        history_indices,
        beliefs,
        batch_size=batch_size,
    )
    truth = frame.loc[eligible, FUTURE_COLUMN].to_numpy(dtype=np.float64)
    current = frame.loc[eligible, "bun_t"].to_numpy(dtype=np.float64)
    subjects = frame.loc[eligible, "subject_id"].astype(str).to_numpy()
    personalized = output["personalized"]
    population = output["population"]
    half_width = output["half_width"]

    personal_error = personalized - truth
    population_error = population - truth
    persistence_error = current - truth
    absolute_error = np.abs(personal_error)
    delta_vs_persistence = absolute_error - np.abs(persistence_error)
    delta_vs_population = absolute_error - np.abs(population_error)
    patient_mae, _ = _patient_equalized(subjects, absolute_error)
    persistence_patient_mae, _ = _patient_equalized(subjects, np.abs(persistence_error))
    population_patient_mae, _ = _patient_equalized(subjects, np.abs(population_error))
    patient_delta_persistence, patient_deltas_persistence = _patient_equalized(
        subjects, delta_vs_persistence
    )
    patient_delta_population, patient_deltas_population = _patient_equalized(
        subjects, delta_vs_population
    )

    actual_delta = truth - current
    predicted_delta = personalized - current
    changed = np.abs(actual_delta) >= 1.0
    direction_accuracy = float(
        np.mean(np.sign(predicted_delta[changed]) == np.sign(actual_delta[changed]))
    ) if changed.any() else float("nan")
    delta_correlation = float(np.corrcoef(predicted_delta, actual_delta)[0, 1])
    coverage = float(
        np.mean((truth >= personalized - half_width) & (truth <= personalized + half_width))
    )
    treatment_columns = [
        column
        for column in loaded_treatment_columns()
        if column in frame.columns
    ]
    treatment_support = {
        column: int(
            (pd.to_numeric(frame.loc[eligible, column], errors="coerce") > 0).sum()
        )
        for column in treatment_columns
    }

    return {
        "schema": "bun_forecast_demo_external_evaluation.v1",
        "artifact": "patient_state_precision_bun3_v1",
        "cohort": str(cohort.resolve()),
        "evaluation": "external_database_model_evaluation",
        "target": TARGET,
        "nominal_horizon_hours": HORIZON_HOURS,
        "history_steps": HISTORY_STEPS,
        "rows": int(len(eligible)),
        "patients": int(pd.Series(subjects).nunique()),
        "stays": int(frame.loc[eligible, "stay_id"].nunique()),
        "metrics": {
            "personalized_mae_mg_dl": float(absolute_error.mean()),
            "personalized_patient_equalized_mae_mg_dl": patient_mae,
            "personalized_rmse_mg_dl": float(np.sqrt(np.mean(personal_error ** 2))),
            "personalized_median_absolute_error_mg_dl": float(np.median(absolute_error)),
            "personalized_bias_mg_dl": float(personal_error.mean()),
            "within_2_mg_dl": float(np.mean(absolute_error <= 2.0)),
            "within_5_mg_dl": float(np.mean(absolute_error <= 5.0)),
            "within_10_mg_dl": float(np.mean(absolute_error <= 10.0)),
            "direction_accuracy_when_change_ge_1_mg_dl": direction_accuracy,
            "delta_correlation": delta_correlation,
            "population_patient_equalized_mae_mg_dl": population_patient_mae,
            "persistence_patient_equalized_mae_mg_dl": persistence_patient_mae,
            "patient_equalized_delta_mae_vs_population_mg_dl": patient_delta_population,
            "patient_equalized_delta_mae_vs_persistence_mg_dl": patient_delta_persistence,
            "delta_vs_population_patient_bootstrap_95_ci": _bootstrap_mean_ci(
                patient_deltas_population, samples=bootstrap_samples
            ),
            "delta_vs_persistence_patient_bootstrap_95_ci": _bootstrap_mean_ci(
                patient_deltas_persistence, samples=bootstrap_samples
            ),
            "interval_nominal_coverage": 0.90,
            "interval_empirical_coverage": coverage,
            "interval_mean_full_width_mg_dl": float(np.mean(2.0 * half_width)),
            "interval_median_full_width_mg_dl": float(np.median(2.0 * half_width)),
        },
        "quality": {
            "requires_six_rows_from_same_stay": True,
            "future_measurements_used_as_inputs": False,
            "future_treatments_used_as_inputs": False,
            "target_tolerance_defined_by_cohort": True,
            "target_tolerance_hours": target_tolerance_hours,
            "observed_treatment_columns": treatment_columns,
            "observed_treatment_positive_windows": treatment_support,
            "history_elapsed_min_hours": float(output["elapsed_min"].min()),
            "history_elapsed_max_hours": float(output["elapsed_max"].max()),
            "patient_identifiers_in_report": False,
        },
        "boundary": {
            "factual_only": True,
            "clinical_claim_allowed": False,
            "causal_claim_allowed": False,
        },
    }


def loaded_treatment_columns() -> tuple[str, ...]:
    return tuple(
        column
        for column in artifact().metadata["feature_columns"]
        if str(column).startswith("hist_")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--target-tolerance-hours", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        args.cohort,
        batch_size=args.batch_size,
        bootstrap_samples=args.bootstrap_samples,
        target_tolerance_hours=args.target_tolerance_hours,
    )
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
