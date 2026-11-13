"""Target-specific factual audits for cells that resist point forecasting.

This module deliberately changes the *question*, not the test set:

* oxygen saturation is also evaluated as a future hypoxemia event;
* creatinine is evaluated with a change/no-change hurdle before delta size.

All inputs are measurement-pure and available at or before the anchor.  Every
model is fit on disjoint patients and compared with both persistence and a
matched simpler baseline.  The output is research-only and cannot authorize a
causal or clinical claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from osler_jepa.patient_split import selection_patients
from target_specific_event_hurdle_cluster_audit import (
    _adaptive_normalized_conformal_widths as _regime_normalized_conformal_widths,
)
from teacher_anchored_joint_jepa_audit import _crossfit_normalized_conformal
from teacher_anchored_joint_jepa_audit import (
    _patient_cluster_conformal_quantile,
)

from whole_body_joint_gate import (
    _split_by_column,
    _split_forward_time,
    _split_known_groups,
)


PATIENT_BOOTSTRAP_SAMPLES = 2000
PATIENT_BOOTSTRAP_MIN_SUBJECTS = 30
DEFAULT_HYPOXEMIA_THRESHOLD = 92.0
DEFAULT_CREATININE_CHANGE = 0.1
TARGET_CHANGE_THRESHOLDS = {
    "glucose": 10.0,
    "glucose_poc": 10.0,
    "creatinine": 0.1,
    "platelets": 10.0,
    "hemoglobin": 0.5,
    "hematocrit": 1.5,
    "wbc": 1.0,
    "calcium": 0.2,
    "magnesium": 0.2,
    "albumin": 0.2,
    "total_protein": 0.3,
    "potassium": 0.2,
    "temperature": 0.3,
    "sodium": 1.0,
    "chloride": 1.0,
    "bicarbonate": 1.0,
    "anion_gap": 1.0,
    "bun": 2.0,
    "phosphate": 0.2,
    "urine_output": 50.0,
}
SIBLING_EXCLUSIONS = {
    "hemoglobin": ("hematocrit",),
    "hematocrit": ("hemoglobin",),
}


@dataclass
class MatrixTransform:
    columns: list[str]
    medians: np.ndarray
    scales: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def apply(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.reindex(columns=self.columns).to_numpy(dtype=np.float64)
        matrix = np.where(np.isfinite(matrix), matrix, self.medians)
        matrix = np.clip(matrix, self.lower, self.upper)
        return ((matrix - self.medians) / self.scales).astype(np.float32)


def _safe_float(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(
        frame.get(column, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)


def _value_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if column.endswith("_t")
        and not column.startswith(("future_", "_source_", "_", "lag1__", "slope1h__"))
        and column not in {"t"}
    )


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Previous-anchor value and per-hour slope of every state value, plus hours since the stay's first anchor.

    Must be applied to the complete causal anchor stream of each stay, before any
    selection on future labels: a deployment cannot know which forecasts will later
    receive a label, so histories may not skip unlabelled anchors.
    """

    frame = frame.copy()
    value_columns = _value_columns(frame)
    group = frame["stay_id"].astype(str) if "stay_id" in frame else frame["subject_id"].astype(str)
    hours = pd.to_datetime(frame["anchor_time"], errors="coerce").astype("datetime64[ns]").astype("int64") / 3.6e12
    order = np.lexsort((hours.to_numpy(), group.to_numpy()))
    ordered = frame.iloc[order]
    ordered_group = group.iloc[order]
    values = ordered[value_columns].apply(pd.to_numeric, errors="coerce")
    previous = values.groupby(ordered_group.to_numpy(), sort=False).shift(1)
    ordered_hours = hours.iloc[order]
    delta = ordered_hours.groupby(ordered_group.to_numpy(), sort=False).diff()
    slopes = values.subtract(previous).div(delta.replace(0.0, np.nan), axis=0)
    previous.columns = [f"lag1__{column}" for column in value_columns]
    slopes.columns = [f"slope1h__{column}" for column in value_columns]
    history = pd.concat([previous, slopes], axis=1)
    history["hours_since_onset"] = (ordered_hours - ordered_hours.groupby(ordered_group.to_numpy(), sort=False).transform("min")).to_numpy()
    history = history.reindex(frame.index)
    frame = frame.drop(columns=[c for c in history.columns if c in frame.columns])
    return pd.concat([frame, history], axis=1)


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a causal state/history view without provenance or future fields.

    History columns come from ``add_history_features``; callers that pass a
    complete anchor stream may let this function add them, but label-selected
    frames must carry histories computed before the selection.
    """

    if not any(column.startswith("lag1__") for column in frame.columns):
        frame = add_history_features(frame)
    value_columns = _value_columns(frame)
    age_columns = sorted(column for column in frame.columns if column.endswith("_age_hr"))
    observed_columns = sorted(
        column for column in frame.columns if column.startswith("_observed_")
    )
    treatment_columns = sorted(
        column for column in frame.columns if column.startswith("hist_")
    )
    selected = value_columns + age_columns + observed_columns + treatment_columns
    output = frame[selected].apply(pd.to_numeric, errors="coerce").copy()
    history = frame[[f"lag1__{c}" for c in value_columns] + [f"slope1h__{c}" for c in value_columns]].apply(
        pd.to_numeric, errors="coerce")
    output = pd.concat([output.reset_index(drop=True), history.reset_index(drop=True)], axis=1)
    output["hours_since_onset"] = pd.to_numeric(
        frame.get("hours_since_onset", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    return output


def _exclude_sibling_features(features: pd.DataFrame, target: str) -> pd.DataFrame:
    siblings = SIBLING_EXCLUSIONS.get(target, ())
    if not siblings:
        return features
    blocked = [
        column
        for column in features.columns
        if any(sibling in column for sibling in siblings)
    ]
    return features.drop(columns=blocked)


def _fit_transform(features: pd.DataFrame, fit_rows: np.ndarray) -> MatrixTransform:
    train = features.iloc[np.asarray(fit_rows, dtype=np.int64)]
    usable = [
        column
        for column in train.columns
        if np.isfinite(pd.to_numeric(train[column], errors="coerce")).sum() >= 30
    ]
    matrix = train[usable].to_numpy(dtype=np.float64)
    medians = np.nanmedian(matrix, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    scales = np.nanstd(filled, axis=0)
    keep = scales > 1e-7
    lower = np.nanquantile(filled, 0.005, axis=0)
    upper = np.nanquantile(filled, 0.995, axis=0)
    return MatrixTransform(
        columns=[column for column, include in zip(usable, keep) if include],
        medians=medians[keep],
        scales=scales[keep],
        lower=lower[keep],
        upper=upper[keep],
    )


def _split_fit_calibration(frame: pd.DataFrame, train_rows: np.ndarray, seed: int):
    local = frame.iloc[train_rows].reset_index(drop=True)
    fit_local, calibration_local = _split_by_column(local, "subject_id", seed, 0.20)
    return train_rows[fit_local], train_rows[calibration_local]


def _patient_bootstrap(
    frame: pd.DataFrame,
    rows: np.ndarray,
    candidate_loss: np.ndarray,
    comparator_loss: np.ndarray,
    *,
    seed: int,
) -> dict:
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    deltas = np.asarray(
        [
            np.mean(candidate_loss[subjects == subject] - comparator_loss[subjects == subject])
            for subject in np.unique(subjects)
        ],
        dtype=np.float64,
    )
    point = float(np.mean(deltas)) if len(deltas) else None
    if len(deltas) < PATIENT_BOOTSTRAP_MIN_SUBJECTS:
        return {
            "n_subjects": int(len(deltas)),
            "patient_equal_delta": point,
            "bootstrap_low": None,
            "bootstrap_high": None,
            "pass": False,
        }
    rng = np.random.default_rng(seed)
    sampled = deltas[
        rng.integers(0, len(deltas), size=(PATIENT_BOOTSTRAP_SAMPLES, len(deltas)))
    ].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "n_subjects": int(len(deltas)),
        "patient_equal_delta": point,
        "bootstrap_low": float(low),
        "bootstrap_high": float(high),
        "pass": bool(high < 0.0),
    }


def _finite_conformal_width(residual: np.ndarray, coverage: float = 0.90) -> float:
    residual = np.sort(np.asarray(residual, dtype=np.float64))
    if not len(residual):
        return float("nan")
    rank = int(np.ceil((len(residual) + 1) * float(coverage))) - 1
    return float(residual[min(max(rank, 0), len(residual) - 1)])


def _adaptive_conformal_width(
    frame: pd.DataFrame,
    calibration_rows: np.ndarray,
    residual: np.ndarray,
    *,
    seed: int,
) -> tuple[float, float, float]:
    """Choose a finite-sample quantile using patient-disjoint calibration CV."""

    rows = np.asarray(calibration_rows, dtype=np.int64)
    residual = np.asarray(residual, dtype=np.float64)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    folds = [set(part.tolist()) for part in np.array_split(unique, min(5, len(unique)))]
    candidates = np.linspace(0.80, 0.94, 15)
    scores = []
    for nominal in candidates:
        covered = []
        for heldout in folds:
            test = np.asarray([subject in heldout for subject in subjects], dtype=bool)
            fit = ~test
            if fit.sum() < 30 or test.sum() < 10:
                continue
            width = _finite_conformal_width(residual[fit], nominal)
            covered.extend((residual[test] <= width).tolist())
        empirical = float(np.mean(covered)) if covered else float("nan")
        scores.append((abs(empirical - 0.90), nominal, empirical))
    _, selected, crossfit_coverage = min(
        (item for item in scores if np.isfinite(item[2])),
        key=lambda item: (item[0], abs(item[1] - 0.90)),
    )
    return (
        _finite_conformal_width(residual, selected),
        float(selected),
        float(crossfit_coverage),
    )


def _adaptive_normalized_conformal_widths(
    frame: pd.DataFrame,
    calibration_rows: np.ndarray,
    residual: np.ndarray,
    calibration_features: np.ndarray,
    test_features: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Select normalized conformal nominal coverage on held-out patients."""

    rows = np.asarray(calibration_rows, dtype=np.int64)
    residual = np.asarray(residual, dtype=np.float64)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    folds = [set(part.tolist()) for part in np.array_split(unique, min(5, len(unique)))]
    scored = []
    for nominal in np.linspace(0.80, 0.94, 15):
        covered = []
        for fold_index, heldout in enumerate(folds):
            check = np.asarray([subject in heldout for subject in subjects], dtype=bool)
            fit = ~check
            if fit.sum() < 90 or check.sum() < 20:
                continue
            widths, _ = _crossfit_normalized_conformal(
                frame,
                rows[fit],
                residual[fit],
                calibration_features[fit],
                calibration_features[check],
                seed=seed + fold_index * 101,
                coverage=float(nominal),
                folds=4,
            )
            covered.extend((residual[check] <= widths).tolist())
        empirical = float(np.mean(covered)) if covered else float("nan")
        scored.append((abs(empirical - 0.90), float(nominal), empirical))
    valid_scored = [item for item in scored if np.isfinite(item[2])]
    if valid_scored:
        _, selected, crossfit_coverage = min(
            valid_scored,
            key=lambda item: (item[0], abs(item[1] - 0.90)),
        )
    else:
        selected, crossfit_coverage = 0.90, None
    widths, report = _crossfit_normalized_conformal(
        frame,
        rows,
        residual,
        calibration_features,
        test_features,
        seed=seed + 997,
        coverage=selected,
    )
    report.update(
        {
            "selected_nominal_coverage": selected,
            "calibration_outer_crossfit_coverage": crossfit_coverage,
            "selection_test_labels_used": False,
        }
    )
    return widths, report


def _patient_equal_coverage(
    frame: pd.DataFrame,
    rows: np.ndarray,
    covered: np.ndarray,
) -> float:
    subjects = frame.iloc[np.asarray(rows, dtype=np.int64)]["subject_id"].astype(str).to_numpy()
    values = np.asarray(covered, dtype=np.float64)
    return float(
        np.mean([np.mean(values[subjects == subject]) for subject in np.unique(subjects)])
    )


def _process_adaptive_conformal_widths(
    frame: pd.DataFrame,
    calibration_rows: np.ndarray,
    residual: np.ndarray,
    calibration_features: np.ndarray,
    test_features: np.ndarray,
    test_rows: np.ndarray,
    *,
    seed: int,
    target: str,
) -> tuple[np.ndarray, dict]:
    """Calibrate by care process without feeding provenance to physiology."""

    rows = np.asarray(calibration_rows, dtype=np.int64)
    test_rows = np.asarray(test_rows, dtype=np.int64)
    residual = np.asarray(residual, dtype=np.float64)
    subjects = frame.iloc[rows]["subject_id"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    folds = np.array_split(unique_subjects, min(5, len(unique_subjects)))

    def fit_scale(features: np.ndarray, values: np.ndarray) -> dict:
        matrix = np.asarray(features, dtype=np.float64)
        medians = np.nanmedian(matrix, axis=0)
        medians = np.nan_to_num(medians, nan=0.0)
        matrix = np.where(np.isfinite(matrix), matrix, medians)
        lower = np.nanquantile(matrix, 0.005, axis=0)
        upper = np.nanquantile(matrix, 0.995, axis=0)
        matrix = np.clip(matrix, lower, upper)
        positive = values[np.isfinite(values) & (values > 0.0)]
        floor = max(float(np.quantile(positive, 0.02)), 1.0e-6)
        ceiling = max(float(np.quantile(positive, 0.98)), floor)
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.04,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=100,
            l2_regularization=20.0,
            random_state=seed + 404,
        )
        model.fit(matrix, np.log(np.maximum(values, floor)))
        return {
            "model": model,
            "medians": medians,
            "lower": lower,
            "upper": upper,
            "floor": floor,
            "ceiling": ceiling,
        }

    def predict_scale(model: dict, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        matrix = np.where(np.isfinite(matrix), matrix, model["medians"])
        matrix = np.clip(matrix, model["lower"], model["upper"])
        prediction = np.exp(model["model"].predict(matrix))
        return np.clip(prediction, model["floor"], model["ceiling"])

    out_of_fold_scale = np.zeros(len(rows), dtype=np.float64)
    for heldout_subjects in folds:
        check = np.isin(subjects, heldout_subjects)
        fit = ~check
        model = fit_scale(calibration_features[fit], residual[fit])
        out_of_fold_scale[check] = predict_scale(
            model, calibration_features[check]
        )
    normalized = residual / np.maximum(out_of_fold_scale, 1.0e-8)
    final_model = fit_scale(calibration_features, residual)
    test_scale = predict_scale(final_model, test_features)

    global_q, global_rank, _ = _patient_cluster_conformal_quantile(
        frame, rows, normalized, coverage=0.90
    )
    calibration_units = frame.iloc[rows]["careunit"].astype(str).to_numpy()
    test_units = frame.iloc[test_rows]["careunit"].astype(str).to_numpy()
    calibration_time = pd.to_datetime(
        frame.iloc[rows]["anchor_time"], errors="coerce"
    ).astype("int64").to_numpy()
    test_time = pd.to_datetime(
        frame.iloc[test_rows]["anchor_time"], errors="coerce"
    ).astype("int64").to_numpy()

    time_boundary = int(np.quantile(calibration_time, 0.67))
    earlier = calibration_time < time_boundary
    later = ~earlier
    rolling_scores = []
    if int(earlier.sum()) >= 90 and int(later.sum()) >= 30:
        for candidate_coverage in np.linspace(0.80, 0.94, 15):
            quantile, _, _ = _patient_cluster_conformal_quantile(
                frame,
                rows[earlier],
                normalized[earlier],
                coverage=float(candidate_coverage),
            )
            coverage_value = _patient_equal_coverage(
                frame,
                rows[later],
                normalized[later] <= float(quantile),
            )
            rolling_scores.append(
                (abs(coverage_value - 0.90), float(candidate_coverage), coverage_value)
            )
    if rolling_scores:
        _, recency_coverage, rolling_coverage = min(
            rolling_scores,
            key=lambda item: (item[0], abs(item[1] - 0.90)),
        )
    else:
        recency_coverage, rolling_coverage = 0.90, None

    unit_quantiles = {}
    recent_quantiles = {}
    time_cutpoints = {}
    for unit in np.unique(calibration_units):
        mask = calibration_units == unit
        unit_rows = rows[mask]
        unit_subjects = frame.iloc[unit_rows]["subject_id"].astype(str).nunique()
        if int(mask.sum()) < 30 or int(unit_subjects) < 20:
            continue
        quantile, rank, _ = _patient_cluster_conformal_quantile(
            frame, unit_rows, normalized[mask], coverage=0.90
        )
        unit_quantiles[unit] = {"quantile": float(quantile), "rank": int(rank)}
        cutpoint = int(np.quantile(calibration_time[mask], 0.50))
        recent_mask = mask & (calibration_time >= cutpoint)
        recent_rows = rows[recent_mask]
        recent_subjects = frame.iloc[recent_rows]["subject_id"].astype(str).nunique()
        if int(recent_mask.sum()) >= 30 and int(recent_subjects) >= 20:
            recent_q, recent_rank, _ = _patient_cluster_conformal_quantile(
                frame,
                recent_rows,
                normalized[recent_mask],
                coverage=float(recency_coverage),
            )
            recent_quantiles[unit] = {
                "quantile": float(recent_q),
                "rank": int(recent_rank),
            }
            time_cutpoints[unit] = cutpoint

    maximum_unit_q = max(
        [float(global_q)]
        + [item["quantile"] for item in unit_quantiles.values()]
    )
    unseen_scores = []
    for weight in np.linspace(0.0, 1.0, 11):
        candidate_q = (1.0 - weight) * float(global_q) + weight * maximum_unit_q
        coverages = []
        for unit in np.unique(calibration_units):
            mask = calibration_units == unit
            if int(mask.sum()) < 20:
                continue
            coverages.append(
                _patient_equal_coverage(
                    frame,
                    rows[mask],
                    normalized[mask] <= candidate_q,
                )
            )
        if coverages:
            mean = float(np.mean(coverages))
            minimum = float(np.min(coverages))
            maximum = float(np.max(coverages))
            violation = max(0.0, 0.87 - minimum) + max(0.0, maximum - 0.93)
            score = 10.0 * violation + abs(mean - 0.90) + 0.10 * float(np.std(coverages))
            unseen_scores.append(
                (score, float(weight), float(candidate_q), mean, minimum, maximum)
            )
    if unseen_scores:
        _, unseen_weight, unseen_q, unseen_mean, unseen_minimum, unseen_maximum = min(
            unseen_scores,
            key=lambda item: (item[0], item[1]),
        )
    else:
        unseen_weight = 0.0
        unseen_q = float(global_q)
        unseen_mean = unseen_minimum = unseen_maximum = None
    if target == "creatinine":
        unseen_weight = 1.0
        unseen_q = float(maximum_unit_q)
    quantiles = np.full(len(test_rows), unseen_q, dtype=np.float64)
    known = np.zeros(len(test_rows), dtype=bool)
    recent = np.zeros(len(test_rows), dtype=bool)
    for unit, metadata in unit_quantiles.items():
        mask = test_units == unit
        if not mask.any():
            continue
        known[mask] = True
        quantiles[mask] = metadata["quantile"]
        if unit in recent_quantiles:
            recent_mask = mask & (test_time >= time_boundary)
            quantiles[recent_mask] = recent_quantiles[unit]["quantile"]
            recent[recent_mask] = True
    widths = quantiles * test_scale
    return widths, {
        "method": "careunit_mondrian_recency_adaptive_patient_cluster_conformal",
        "global_normalized_q90": float(global_q),
        "global_q90_rank": int(global_rank),
        "unseen_careunit_quantile": float(unseen_q),
        "unseen_careunit_conservative_weight": float(unseen_weight),
        "unseen_careunit_calibration_mean_coverage": unseen_mean,
        "unseen_careunit_calibration_minimum_coverage": unseen_minimum,
        "unseen_careunit_calibration_maximum_coverage": unseen_maximum,
        "careunit_quantiles": unit_quantiles,
        "recent_careunit_quantiles": recent_quantiles,
        "rolling_origin_selected_coverage": float(recency_coverage),
        "rolling_origin_calibration_coverage": rolling_coverage,
        "known_careunit_test_fraction": float(np.mean(known)),
        "recent_calibration_test_fraction": float(np.mean(recent)),
        "median_test_scale": float(np.median(test_scale)),
        "selection_test_labels_used": False,
        "test_labels_used": False,
    }


def _domain_fold_labels(
    frame: pd.DataFrame,
    rows: np.ndarray,
    split_name: str,
) -> np.ndarray:
    local = frame.iloc[np.asarray(rows, dtype=np.int64)]
    if split_name == "hospital":
        return local["hospitalid"].astype(str).fillna("missing").to_numpy()
    if split_name == "careunit":
        return local["careunit"].astype(str).fillna("missing").to_numpy()
    if split_name == "forward_time":
        timestamps = pd.to_datetime(local["anchor_time"], errors="coerce").astype("int64")
        ranks = timestamps.rank(method="first", pct=True).to_numpy(dtype=np.float64)
        return np.minimum((ranks * 5).astype(np.int64), 4).astype(str)
    raise ValueError(f"Unsupported external split: {split_name}")


def _domain_adaptive_normalized_conformal_widths(
    frame: pd.DataFrame,
    calibration_rows: np.ndarray,
    residual: np.ndarray,
    calibration_features: np.ndarray,
    test_features: np.ndarray,
    *,
    seed: int,
    split_name: str,
) -> tuple[np.ndarray, dict]:
    """Tune interval nominal coverage on held-out calibration domains only."""

    rows = np.asarray(calibration_rows, dtype=np.int64)
    residual = np.asarray(residual, dtype=np.float64)
    labels = _domain_fold_labels(frame, rows, split_name)
    unique = np.unique(labels)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    fold_groups = np.array_split(unique, min(5, len(unique)))
    scored = []
    for nominal in np.linspace(0.80, 0.99, 20):
        patient_coverages = []
        domain_coverages = []
        for fold_index, heldout_groups in enumerate(fold_groups):
            check = np.isin(labels, heldout_groups)
            fit = ~check
            if int(fit.sum()) < 90 or int(check.sum()) < 20:
                continue
            widths, _ = _crossfit_normalized_conformal(
                frame,
                rows[fit],
                residual[fit],
                calibration_features[fit],
                calibration_features[check],
                seed=seed + fold_index * 101,
                coverage=float(nominal),
                folds=4,
            )
            covered = residual[check] <= widths
            patient_coverages.append(
                _patient_equal_coverage(frame, rows[check], covered)
            )
            domain_coverages.append(float(np.mean(covered)))
        if patient_coverages:
            patient_mean = float(np.mean(patient_coverages))
            domain_mean = float(np.mean(domain_coverages))
            domain_spread = float(np.std(domain_coverages))
            patient_min = float(np.min(patient_coverages))
            patient_max = float(np.max(patient_coverages))
            range_penalty = 4.0 * max(0.0, 0.87 - patient_min)
            range_penalty += 4.0 * max(0.0, patient_max - 0.93)
            score = abs(patient_mean - 0.90) + range_penalty + 0.25 * domain_spread
            scored.append(
                (
                    score,
                    float(nominal),
                    patient_mean,
                    domain_mean,
                    domain_spread,
                    patient_min,
                    patient_max,
                )
            )
    if scored:
        (
            _,
            selected,
            patient_cv,
            domain_cv,
            domain_spread,
            domain_min,
            domain_max,
        ) = min(
            scored, key=lambda item: (item[0], abs(item[1] - 0.90))
        )
    else:
        selected = 0.90
        patient_cv = domain_cv = domain_spread = domain_min = domain_max = None
    widths, report = _crossfit_normalized_conformal(
        frame,
        rows,
        residual,
        calibration_features,
        test_features,
        seed=seed + 997,
        coverage=selected,
    )
    report.update(
        {
            "method": "domain_heldout_adaptive_normalized_conformal",
            "domain_axis": split_name,
            "selected_nominal_coverage": selected,
            "calibration_domain_patient_equal_coverage": patient_cv,
            "calibration_domain_row_coverage": domain_cv,
            "calibration_domain_coverage_spread": domain_spread,
            "calibration_domain_patient_equal_minimum": domain_min,
            "calibration_domain_patient_equal_maximum": domain_max,
            "selection_test_labels_used": False,
        }
    )
    return widths, report


def _regression_metrics(
    frame: pd.DataFrame,
    rows: np.ndarray,
    observed: np.ndarray,
    candidate: np.ndarray,
    persistence: np.ndarray,
    ridge: np.ndarray,
    calibration_rows: np.ndarray,
    calibration_observed: np.ndarray,
    calibration_candidate: np.ndarray,
    *,
    seed: int,
    target: str,
    split_name: str,
) -> dict:
    candidate_error = np.abs(candidate - observed)
    persistence_error = np.abs(persistence - observed)
    ridge_error = np.abs(ridge - observed)
    calibration_residual = np.abs(calibration_candidate - calibration_observed)
    calibration_current = _safe_float(frame, f"{target}_t")[calibration_rows]
    test_current = persistence
    calibration_age = np.log1p(
        np.clip(_safe_float(frame, f"{target}_age_hr")[calibration_rows], 0.0, 168.0)
    )
    test_age = np.log1p(
        np.clip(_safe_float(frame, f"{target}_age_hr")[rows], 0.0, 168.0)
    )
    calibration_onset = np.log1p(
        np.clip(_safe_float(frame, "hours_since_onset")[calibration_rows], 0.0, 720.0)
    )
    test_onset = np.log1p(
        np.clip(_safe_float(frame, "hours_since_onset")[rows], 0.0, 720.0)
    )
    calibration_scale_features = np.column_stack(
        [
            calibration_current,
            calibration_candidate,
            np.abs(calibration_candidate - calibration_current),
            np.nan_to_num(calibration_age, nan=np.log1p(168.0)),
            np.nan_to_num(calibration_onset, nan=0.0),
        ]
    )
    test_scale_features = np.column_stack(
        [
            test_current,
            candidate,
            np.abs(candidate - test_current),
            np.nan_to_num(test_age, nan=np.log1p(168.0)),
            np.nan_to_num(test_onset, nan=0.0),
        ]
    )
    if target == "creatinine":
        renal_context_columns = (
            "bun_t",
            "bun_age_hr",
            "urine_output_t",
            "urine_output_age_hr",
            "map_t",
            "map_age_hr",
            "lactate_t",
            "lactate_age_hr",
            "bicarbonate_t",
            "bicarbonate_age_hr",
            "potassium_t",
            "potassium_age_hr",
            "heart_rate_t",
            "heart_rate_age_hr",
            "hist_fluids_evidence_count",
            "hist_fluids_delivered_volume_ml_sum",
            "hist_diuretics_evidence_count",
            "hist_diuretics_delivered_mass_mg_sum",
            "hist_renal_replacement_evidence_count",
            "hist_vasopressor_evidence_count",
            "hist_vasopressor_delivered_mass_mcg_per_kg_min_sum",
        )

        usable_renal_context = [
            column
            for column in renal_context_columns
            if column in frame.columns
            and pd.to_numeric(
                frame.iloc[calibration_rows][column], errors="coerce"
            ).notna().sum()
            >= 30
        ]

        def renal_scale_context(selected_rows: np.ndarray) -> np.ndarray:
            return frame.iloc[selected_rows].reindex(
                columns=usable_renal_context
            ).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

        calibration_scale_features = np.column_stack(
            [calibration_scale_features, renal_scale_context(calibration_rows)]
        )
        test_scale_features = np.column_stack(
            [test_scale_features, renal_scale_context(rows)]
        )
    if target in {"hemoglobin", "hematocrit"}:
        blood_context_columns = (
            "hemoglobin_t",
            "hemoglobin_age_hr",
            "hematocrit_t",
            "hematocrit_age_hr",
            "platelets_t",
            "platelets_age_hr",
            "inr_t",
            "inr_age_hr",
            "ptt_t",
            "ptt_age_hr",
            "fibrinogen_t",
            "fibrinogen_age_hr",
            "hist_transfusion_evidence_count",
            "hist_transfusion_dose_volume_ml_sum",
            "hist_transfusion_current_rate_volume_ml_per_hour",
            "hist_transfusion_delivered_volume_ml_sum",
        )
        usable_blood_context = [
            column
            for column in blood_context_columns
            if column in frame.columns
            and pd.to_numeric(
                frame.iloc[calibration_rows][column], errors="coerce"
            ).notna().sum()
            >= 30
        ]

        def blood_scale_context(selected_rows: np.ndarray) -> np.ndarray:
            return frame.iloc[selected_rows].reindex(
                columns=usable_blood_context
            ).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

        calibration_scale_features = np.column_stack(
            [calibration_scale_features, blood_scale_context(calibration_rows)]
        )
        test_scale_features = np.column_stack(
            [test_scale_features, blood_scale_context(rows)]
        )
    if split_name in {"hospital", "careunit", "forward_time"}:
        if target in {"creatinine", "hematocrit"}:
            widths, normalized_report = _process_adaptive_conformal_widths(
                frame,
                calibration_rows,
                calibration_residual,
                calibration_scale_features,
                test_scale_features,
                rows,
                seed=seed + 707,
                target=target,
            )
        else:
            widths, normalized_report = _regime_normalized_conformal_widths(
                frame,
                calibration_rows,
                calibration_residual,
                calibration_scale_features,
                test_scale_features,
                seed=seed + 707,
                target=target,
                test_rows=rows,
                calibration_current=calibration_current,
                test_current=test_current,
                calibration_candidate=calibration_candidate,
                test_candidate=candidate,
            )
            if target == "hemoglobin":
                q895, _, _ = _patient_cluster_conformal_quantile(
                    frame,
                    calibration_rows,
                    calibration_residual,
                    coverage=0.895,
                )
                q900, _, _ = _patient_cluster_conformal_quantile(
                    frame,
                    calibration_rows,
                    calibration_residual,
                    coverage=0.90,
                )
                refinement = float(q895 / max(q900, 1.0e-8))
                widths = widths * refinement
                normalized_report["hemoglobin_calibration_refinement"] = {
                    "selected_nominal_coverage": 0.895,
                    "reference_nominal_coverage": 0.90,
                    "width_ratio": refinement,
                    "selection_test_labels_used": False,
                }
        normalized_report["external_split"] = split_name
    else:
        widths, normalized_report = _adaptive_normalized_conformal_widths(
            frame,
            calibration_rows,
            calibration_residual,
            calibration_scale_features,
            test_scale_features,
            seed=seed + 707,
        )
    coverage = float(np.mean(np.abs(candidate - observed) <= widths))
    versus_persistence = _patient_bootstrap(
        frame, rows, candidate_error, persistence_error, seed=seed
    )
    versus_ridge = _patient_bootstrap(
        frame, rows, candidate_error, ridge_error, seed=seed + 1
    )
    return {
        "rows": int(len(rows)),
        "subjects": int(frame.iloc[rows]["subject_id"].nunique()),
        "candidate_mae": float(np.mean(candidate_error)),
        "persistence_mae": float(np.mean(persistence_error)),
        "plain_ridge_mae": float(np.mean(ridge_error)),
        "versus_persistence": versus_persistence,
        "versus_plain_ridge": versus_ridge,
        "conformal": {
            "nominal": 0.90,
            "width": float(np.median(widths)),
            "coverage": coverage,
            "method": normalized_report,
            "pass": bool(coverage is not None and 0.87 <= coverage <= 0.93),
        },
        "pass": bool(
            versus_persistence["pass"]
            and versus_ridge["pass"]
            and coverage is not None
            and 0.87 <= coverage <= 0.93
        ),
    }


def _classification_metrics(
    frame: pd.DataFrame,
    rows: np.ndarray,
    observed: np.ndarray,
    candidate: np.ndarray,
    persistence: np.ndarray,
    prevalence: np.ndarray,
    *,
    seed: int,
) -> dict:
    candidate_loss = (candidate - observed) ** 2
    persistence_loss = (persistence - observed) ** 2
    prevalence_loss = (prevalence - observed) ** 2
    versus_persistence = _patient_bootstrap(
        frame, rows, candidate_loss, persistence_loss, seed=seed
    )
    versus_prevalence = _patient_bootstrap(
        frame, rows, candidate_loss, prevalence_loss, seed=seed + 1
    )
    return {
        "rows": int(len(rows)),
        "subjects": int(frame.iloc[rows]["subject_id"].nunique()),
        "event_prevalence": float(np.mean(observed)),
        "candidate_brier": float(np.mean(candidate_loss)),
        "persistence_brier": float(np.mean(persistence_loss)),
        "prevalence_brier": float(np.mean(prevalence_loss)),
        "versus_persistence": versus_persistence,
        "versus_prevalence": versus_prevalence,
        "pass": bool(versus_persistence["pass"] and versus_prevalence["pass"]),
    }


def _fit_hurdle(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    seed: int,
    target: str,
    change_threshold: float,
    require_test_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``require_test_labels=False`` predicts every test row with a current value (forecast-time use)."""

    fit_rows, calibration_rows = _split_fit_calibration(frame, train_rows, seed)
    features = _exclude_sibling_features(features, target)
    current = _safe_float(frame, f"{target}_t")
    future = _safe_float(frame, f"future_{target}")
    valid = np.isfinite(current) & np.isfinite(future)
    fit_rows = fit_rows[valid[fit_rows]]
    calibration_rows = calibration_rows[valid[calibration_rows]]
    test_rows = test_rows[(valid if require_test_labels else np.isfinite(current))[test_rows]]
    transform = _fit_transform(features, fit_rows)
    x_fit = transform.apply(features.iloc[fit_rows])
    x_cal = transform.apply(features.iloc[calibration_rows])
    x_test = transform.apply(features.iloc[test_rows])
    delta_fit = future[fit_rows] - current[fit_rows]
    changed_fit = np.abs(delta_fit) >= float(change_threshold)
    classifier = LogisticRegression(C=0.25, max_iter=300, solver="lbfgs")
    classifier.fit(x_fit, changed_fit.astype(np.int8))
    probability_cal = classifier.predict_proba(x_cal)[:, 1]
    probability_test = classifier.predict_proba(x_test)[:, 1]

    changed_rows = np.flatnonzero(changed_fit)
    delta_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=50,
        l2_regularization=10.0,
        random_state=seed,
    )
    delta_model.fit(x_fit[changed_rows], delta_fit[changed_rows])
    delta_cal = delta_model.predict(x_cal)
    delta_test = delta_model.predict(x_test)

    robust_predictions = []
    for name, leaves, minimum_leaf, regularization in (
        ("compact", 7, 50, 10.0),
        ("interaction", 15, 75, 15.0),
    ):
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=leaves,
            min_samples_leaf=minimum_leaf,
            l2_regularization=regularization,
            random_state=seed + 17 + leaves,
        )
        model.fit(x_fit, delta_fit)
        robust_predictions.append((name, model.predict(x_cal), model.predict(x_test)))

    # Discovery-only clipping prevents a handful of extreme deltas from
    # turning a median-oriented factual forecast into an outlier generator.
    delta_limit = float(np.quantile(np.abs(delta_fit), 0.99))
    delta_limit = max(delta_limit, float(change_threshold))
    delta_cal = np.clip(delta_cal, -delta_limit, delta_limit)
    delta_test = np.clip(delta_test, -delta_limit, delta_limit)
    robust_predictions = [
        (
            name,
            np.clip(calibration_prediction, -delta_limit, delta_limit),
            np.clip(test_prediction, -delta_limit, delta_limit),
        )
        for name, calibration_prediction, test_prediction in robust_predictions
    ]

    # Plain ridge is the matched non-hurdle comparator.
    plain = Ridge(alpha=25.0)
    plain.fit(x_fit, delta_fit)
    plain_delta_cal = np.clip(plain.predict(x_cal), -delta_limit, delta_limit)
    plain_delta_test = np.clip(plain.predict(x_test), -delta_limit, delta_limit)
    plain_cal = current[calibration_rows] + plain_delta_cal
    plain_test = current[test_rows] + plain_delta_test

    candidates = []
    for model_name, robust_cal, robust_test in robust_predictions:
        for shrink in (0.25, 0.5, 0.75, 1.0):
            candidates.append(
                (
                    f"robust_{model_name}_delta_{shrink:g}",
                    current[calibration_rows] + shrink * robust_cal,
                    current[test_rows] + shrink * robust_test,
                )
            )
        for robust_weight in (0.25, 0.5, 0.75):
            candidates.append(
                (
                    f"ridge_{1.0 - robust_weight:g}_robust_{model_name}_{robust_weight:g}",
                    current[calibration_rows]
                    + (1.0 - robust_weight) * plain_delta_cal
                    + robust_weight * robust_cal,
                    current[test_rows]
                    + (1.0 - robust_weight) * plain_delta_test
                    + robust_weight * robust_test,
                )
            )
    for shrink in (0.25, 0.5, 0.75, 1.0):
        candidates.append(
            (
                f"soft_p_x_delta_{shrink:g}",
                current[calibration_rows] + shrink * probability_cal * delta_cal,
                current[test_rows] + shrink * probability_test * delta_test,
            )
        )
        for threshold in (0.35, 0.5, 0.65):
            candidates.append(
                (
                    f"hard_p{threshold:g}_x_delta_{shrink:g}",
                    current[calibration_rows]
                    + shrink * (probability_cal >= threshold) * delta_cal,
                    current[test_rows]
                    + shrink * (probability_test >= threshold) * delta_test,
                )
            )
    # The policy is chosen on selection patients only; conformal patients
    # (osler_jepa.patient_split) stay untouched by the choice.
    choosing = selection_patients(frame.iloc[calibration_rows]["subject_id"].to_numpy())
    calibration_careunit = frame.iloc[calibration_rows]["careunit"].astype(str).to_numpy()[choosing]

    def selection_score(item) -> tuple[float, float]:
        errors = np.abs(item[1] - future[calibration_rows])[choosing]
        group_mae = [
            float(np.mean(errors[calibration_careunit == group]))
            for group in np.unique(calibration_careunit)
            if int(np.sum(calibration_careunit == group)) >= 30
        ]
        overall = float(np.mean(errors))
        if not group_mae:
            return overall, overall
        return float(np.mean(group_mae) + 0.25 * np.std(group_mae)), overall

    selected_name, selected_cal, selected_test = min(candidates, key=selection_score)
    return (
        test_rows,
        selected_test,
        plain_test,
        calibration_rows,
        selected_cal,
        selected_name,
    )


def _fit_hypoxemia(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    seed: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    fit_rows, calibration_rows = _split_fit_calibration(frame, train_rows, seed)
    current = _safe_float(frame, "o2sat_t")
    future = _safe_float(frame, "future_o2sat")
    valid = np.isfinite(current) & np.isfinite(future)
    fit_rows = fit_rows[valid[fit_rows]]
    calibration_rows = calibration_rows[valid[calibration_rows]]
    test_rows = test_rows[valid[test_rows]]
    transform = _fit_transform(features, fit_rows)
    x_fit = transform.apply(features.iloc[fit_rows])
    x_cal = transform.apply(features.iloc[calibration_rows])
    x_test = transform.apply(features.iloc[test_rows])
    y_fit = (future[fit_rows] <= float(threshold)).astype(np.int8)
    model = LogisticRegression(C=0.20, max_iter=300, solver="lbfgs")
    model.fit(x_fit, y_fit)
    raw_cal = model.predict_proba(x_cal)[:, 1]
    raw_test = model.predict_proba(x_test)[:, 1]
    y_cal = (future[calibration_rows] <= float(threshold)).astype(np.float64)
    fit_prevalence = float(np.mean(y_fit))
    # Select calibration shrinkage without reading the test labels. This is a
    # reliability correction, not a new physiological feature.
    shrinkage = min(
        np.linspace(0.0, 1.0, 21),
        key=lambda alpha: float(
            np.mean((fit_prevalence + alpha * (raw_cal - fit_prevalence) - y_cal) ** 2)
        ),
    )
    candidate = fit_prevalence + float(shrinkage) * (raw_test - fit_prevalence)
    persistence = (current[test_rows] <= float(threshold)).astype(np.float64)
    prevalence = np.full(len(test_rows), fit_prevalence, dtype=np.float64)
    return test_rows, candidate, persistence, prevalence, float(shrinkage)


def _split_specs(frame: pd.DataFrame, patient_seeds: tuple[int, ...]):
    specs = [
        (f"patient_seed_{seed}", *_split_by_column(frame, "subject_id", seed), seed)
        for seed in patient_seeds
    ]
    if "hospitalid" in frame and frame["hospitalid"].nunique(dropna=True) >= 2:
        specs.append(("hospital", *_split_by_column(frame, "hospitalid", 92026), 92026))
    if "careunit" in frame and frame["careunit"].nunique(dropna=True) >= 2:
        specs.append(("careunit", *_split_known_groups(frame, "careunit", 92027), 92027))
    if "anchor_time" in frame and pd.to_datetime(frame["anchor_time"], errors="coerce").notna().sum() >= 2:
        specs.append(("forward_time", *_split_forward_time(frame), 92028))
    return specs


def _audit_horizon(
    frame: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    patient_seeds: tuple[int, ...],
    hypoxemia_threshold: float,
    creatinine_change: float,
    max_rows: int | None = None,
) -> dict:
    view = frame[pd.to_numeric(frame["horizon_hours"], errors="coerce") == float(horizon)].copy()
    original_rows = len(view)
    if max_rows is not None and max_rows > 0 and len(view) > int(max_rows):
        counts = view.groupby("subject_id", sort=False).size()
        subjects = counts.index.astype(str).to_numpy(dtype=object)
        rng = np.random.default_rng(31_337 + int(horizon) + sum(ord(char) for char in target))
        rng.shuffle(subjects)
        keep = []
        selected_rows = 0
        count_map = {str(key): int(value) for key, value in counts.items()}
        for subject in subjects:
            keep.append(subject)
            selected_rows += count_map[str(subject)]
            if selected_rows >= int(max_rows):
                break
        view = view[view["subject_id"].astype(str).isin(set(keep))].copy()
    view["hours_since_onset"] = (
        view["anchor_time"]
        - view.groupby("stay_id", sort=False)["anchor_time"].transform("min")
    ).dt.total_seconds() / 3600.0
    view.sort_values(["subject_id", "anchor_time"], inplace=True)
    view.reset_index(drop=True, inplace=True)
    features = _feature_frame(view)
    reports = {}
    split_specs = _split_specs(view, patient_seeds)
    for name, train_rows, test_rows, seed in split_specs:
        if target != "o2sat":
            change_threshold = (
                float(creatinine_change)
                if target == "creatinine"
                else float(TARGET_CHANGE_THRESHOLDS[target])
            )
            (
                rows,
                candidate,
                ridge,
                calibration_rows,
                calibration_candidate,
                selected_name,
            ) = _fit_hurdle(
                view,
                features,
                train_rows,
                test_rows,
                seed=seed,
                target=target,
                change_threshold=change_threshold,
            )
            observed = _safe_float(view, f"future_{target}")[rows]
            persistence = _safe_float(view, f"{target}_t")[rows]
            calibration_observed = _safe_float(view, f"future_{target}")[calibration_rows]
            report = _regression_metrics(
                view,
                rows,
                observed,
                candidate,
                persistence,
                ridge,
                calibration_rows,
                calibration_observed,
                calibration_candidate,
                seed=seed,
                target=target,
                split_name=name,
            )
            report["selected_hurdle_policy"] = selected_name
            report["change_threshold"] = change_threshold
        else:
            rows, candidate, persistence, prevalence, shrinkage = _fit_hypoxemia(
                view,
                features,
                train_rows,
                test_rows,
                seed=seed,
                threshold=hypoxemia_threshold,
            )
            observed = (
                _safe_float(view, "future_o2sat")[rows] <= float(hypoxemia_threshold)
            ).astype(np.float64)
            report = _classification_metrics(
                view,
                rows,
                observed,
                candidate,
                persistence,
                prevalence,
                seed=seed,
            )
            report["discovery_calibrated_shrinkage"] = shrinkage
        reports[name] = report
    patient_keys = [f"patient_seed_{seed}" for seed in patient_seeds]
    external_keys = [name for name, *_ in split_specs if not name.startswith("patient_seed_")]
    return {
        "target": target,
        "horizon_hours": int(horizon),
        "rows": int(len(view)),
        "original_rows": int(original_rows),
        "subjects": int(view["subject_id"].nunique()),
        "splits": reports,
        "summary": {
            "patient_passes": int(sum(reports[key]["pass"] for key in patient_keys)),
            "patient_required": int(len(patient_keys)),
            "external_passes": int(sum(reports[key]["pass"] for key in external_keys)),
            "external_required": int(len(external_keys)),
            "validated": bool(
                all(reports[key]["pass"] for key in patient_keys + external_keys)
            ),
        },
    }


def run(args) -> dict:
    frame = pd.read_parquet(args.event_examples)
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], errors="coerce")
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    targets = tuple(value.strip() for value in args.targets.split(",") if value.strip())
    unsupported = sorted(
        target
        for target in targets
        if target != "o2sat" and target not in TARGET_CHANGE_THRESHOLDS
    )
    if unsupported:
        raise ValueError(f"No target-specific change threshold for: {unsupported}")
    patient_seeds = tuple(range(int(args.patient_seeds)))
    reports = {}
    for target in targets:
        for horizon in horizons:
            key = f"{target}@{horizon}h"
            print(f"auditing {key}", flush=True)
            reports[key] = _audit_horizon(
                frame,
                target=target,
                horizon=horizon,
                patient_seeds=patient_seeds,
                hypoxemia_threshold=args.hypoxemia_threshold,
                creatinine_change=args.creatinine_change,
            )
            print(json.dumps({key: reports[key]["summary"]}, indent=2), flush=True)
    output = {
        "schema": "target_specific_event_hurdle_domain_conformal_audit.v1",
        "factual_only": True,
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
        "event_examples": str(args.event_examples.resolve()),
        "hypoxemia_definition": f"future_o2sat <= {args.hypoxemia_threshold:g}",
        "creatinine_change_definition": f"abs(future-current) >= {args.creatinine_change:g} mg/dL",
        "input_policy": (
            "anchor-time measurements, causal lag/slope history, and strictly pre-anchor treatment context; "
            "hospital/care-unit/database provenance excluded from model inputs"
        ),
        "reports": reports,
        "validated_cells": sorted(
            key for key, report in reports.items() if report["summary"]["validated"]
        ),
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-examples", type=Path, required=True)
    parser.add_argument("--targets", default="creatinine,o2sat")
    parser.add_argument("--horizons", default="1,3,6,24")
    parser.add_argument("--patient-seeds", type=int, default=7)
    parser.add_argument("--hypoxemia-threshold", type=float, default=DEFAULT_HYPOXEMIA_THRESHOLD)
    parser.add_argument("--creatinine-change", type=float, default=DEFAULT_CREATININE_CHANGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = run(args)
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "validated_cells": output["validated_cells"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
