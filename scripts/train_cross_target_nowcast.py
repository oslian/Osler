"""Train eICU cross-target nowcasts and externally gate them on MIMIC-IV.

The target is hidden completely. Only other current physiological values are
used. MIMIC-IV is split by patient into calibration and evaluation sets; model
selection and point-model fitting never use MIMIC outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from whole_body_jepa.contracts import VARIABLE_BOUNDS, VARIABLE_UNITS
from whole_body_jepa.nowcast import measurement_pure_feature_columns


SEEDS = (7, 11, 19, 23, 37, 53, 71)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _patient_partition(values: pd.Series, seed: int) -> np.ndarray:
    def bucket(value: object) -> int:
        digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % 2
    return values.map(bucket).to_numpy(dtype=np.int8)


def _patient_errors(rows: pd.DataFrame, prediction: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    truth = rows["truth"].to_numpy(dtype=np.float64)
    delta = np.abs(prediction - truth) - np.abs(baseline - truth)
    return (
        pd.DataFrame({"subject_id": rows["subject_id"].to_numpy(), "delta": delta})
        .groupby("subject_id", sort=False)["delta"]
        .mean()
        .to_numpy(dtype=np.float64)
    )


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _evaluate(rows: pd.DataFrame, prediction: np.ndarray, baseline: np.ndarray, placebo: np.ndarray, seed: int, samples: int) -> dict[str, object]:
    truth = rows["truth"].to_numpy(dtype=np.float64)
    candidate_delta = _patient_errors(rows, prediction, baseline)
    placebo_delta = _patient_errors(rows, prediction, placebo)
    candidate_ci = _bootstrap_ci(candidate_delta, seed, samples)
    placebo_ci = _bootstrap_ci(placebo_delta, seed + 1000, samples)
    return {
        "rows": int(len(rows)),
        "subjects": int(rows["subject_id"].nunique()),
        "mae": {
            "cross_target": float(np.mean(np.abs(prediction - truth))),
            "population_median": float(np.mean(np.abs(baseline - truth))),
            "capacity_placebo": float(np.mean(np.abs(placebo - truth))),
        },
        "delta_vs_median": float(candidate_delta.mean()),
        "delta_vs_median_ci95": candidate_ci,
        "delta_vs_placebo": float(placebo_delta.mean()),
        "delta_vs_placebo_ci95": placebo_ci,
        "point_gate_passed": bool(candidate_ci[1] < 0.0 and placebo_ci[1] < 0.0),
    }


def _fit_model(frame: pd.DataFrame, features: tuple[str, ...], target: str, alpha: float, placebo: bool = False):
    rows = frame[pd.to_numeric(frame[f"{target}_t"], errors="coerce").notna()].copy()
    y = pd.to_numeric(rows[f"{target}_t"], errors="coerce").to_numpy(dtype=np.float64)
    if placebo:
        y = np.random.default_rng(20260827).permutation(y)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        Ridge(alpha=alpha),
    )
    model.fit(rows[list(features)], y)
    return model, rows


def _care_unit_gate(rows: pd.DataFrame, prediction: np.ndarray, baseline: np.ndarray, placebo: np.ndarray, minimum_subjects: int) -> dict[str, object]:
    unit_column = "first_careunit" if "first_careunit" in rows else "careunit"
    reports: dict[str, object] = {}
    supported_subjects = 0
    passed_subjects = 0
    for unit, positions in rows.groupby(unit_column, dropna=False, sort=True).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        subset = rows.iloc[positions]
        if subset["subject_id"].nunique() < minimum_subjects:
            continue
        truth = subset["truth"].to_numpy(dtype=np.float64)
        candidate_mae = float(np.mean(np.abs(prediction[positions] - truth)))
        baseline_mae = float(np.mean(np.abs(baseline[positions] - truth)))
        placebo_mae = float(np.mean(np.abs(placebo[positions] - truth)))
        unit_passed = candidate_mae < baseline_mae and candidate_mae < placebo_mae
        reports[str(unit)] = {
            "subjects": int(subset["subject_id"].nunique()),
            "cross_target_mae": candidate_mae,
            "population_median_mae": baseline_mae,
            "capacity_placebo_mae": placebo_mae,
            "passed": unit_passed,
        }
        supported_subjects += int(subset["subject_id"].nunique())
        if unit_passed:
            passed_subjects += int(subset["subject_id"].nunique())
    major_threshold = max(minimum_subjects, int(np.ceil(0.10 * supported_subjects)))
    major_regressions = [
        unit for unit, payload in reports.items()
        if payload["subjects"] >= major_threshold and not payload["passed"]
    ]
    pass_fraction = passed_subjects / supported_subjects if supported_subjects else 0.0
    return {
        "passed": bool(reports) and pass_fraction >= 0.90 and not major_regressions,
        "patient_weighted_pass_fraction": float(pass_fraction),
        "major_unit_subject_threshold": int(major_threshold),
        "major_unit_regressions": major_regressions,
        "units": reports,
    }


def _target_rows(frame: pd.DataFrame, target: str, max_truth_age: float) -> pd.DataFrame:
    value = pd.to_numeric(frame.get(f"{target}_t"), errors="coerce")
    age = pd.to_numeric(frame.get(f"{target}_age_hr"), errors="coerce")
    rows = frame[value.notna() & age.notna() & age.le(max_truth_age)].copy()
    rows["truth"] = pd.to_numeric(rows[f"{target}_t"], errors="coerce")
    return rows.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eicu", type=Path, required=True)
    parser.add_argument("--mimic", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--max-truth-age-hours", type=float, default=1.0)
    parser.add_argument("--min-eicu-rows", type=int, default=1000)
    parser.add_argument("--min-mimic-subjects", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eicu = pd.read_parquet(args.eicu)
    mimic = pd.read_parquet(args.mimic)
    variables = tuple(
        variable for variable in VARIABLE_UNITS
        if f"{variable}_t" in eicu and f"{variable}_t" in mimic
    )
    reports: dict[str, object] = {}
    heads: dict[str, object] = {}
    for target in variables:
        features = measurement_pure_feature_columns(
            set(eicu.columns) & set(mimic.columns), variables, target
        )
        eicu_rows = _target_rows(eicu, target, args.max_truth_age_hours)
        mimic_rows = _target_rows(mimic, target, args.max_truth_age_hours)
        if len(eicu_rows) < args.min_eicu_rows or mimic_rows["subject_id"].nunique() < args.min_mimic_subjects or not features:
            reports[target] = {
                "status": "insufficient_support",
                "eicu_rows": int(len(eicu_rows)),
                "mimic_subjects": int(mimic_rows["subject_id"].nunique()),
                "feature_count": len(features),
            }
            continue
        model, _ = _fit_model(eicu_rows, features, target, args.alpha)
        placebo_model, _ = _fit_model(eicu_rows, features, target, args.alpha, placebo=True)
        eicu_median = float(np.median(eicu_rows["truth"]))
        seed_reports = []
        final_payload = None
        for seed in SEEDS:
            partition = _patient_partition(mimic_rows["subject_id"], seed)
            calibration = mimic_rows[partition == 0].copy()
            test = mimic_rows[partition == 1].copy()
            if calibration.empty or test.empty:
                continue
            prediction = model.predict(test[list(features)])
            placebo = placebo_model.predict(test[list(features)])
            baseline = np.full(len(test), eicu_median)
            evaluation = _evaluate(test, prediction, baseline, placebo, seed, args.bootstrap_samples)
            calibration_prediction = model.predict(calibration[list(features)])
            radius = float(np.quantile(np.abs(calibration_prediction - calibration["truth"].to_numpy()), 0.9, method="higher"))
            truth = test["truth"].to_numpy(dtype=np.float64)
            coverage = float(np.mean((truth >= prediction - radius) & (truth <= prediction + radius)))
            evaluation["interval_radius"] = radius
            evaluation["interval_coverage"] = coverage
            evaluation["interval_gate_passed"] = bool(0.87 <= coverage <= 0.93)
            evaluation["care_unit"] = _care_unit_gate(
                test, prediction, baseline, placebo, minimum_subjects=50
            )
            evaluation["seed"] = seed
            seed_reports.append(evaluation)
            if seed == SEEDS[0]:
                final_payload = (radius, coverage)
        point_passed = bool(seed_reports) and all(
            row["point_gate_passed"] and row["care_unit"]["passed"] for row in seed_reports
        )
        interval_passed = point_passed and all(row["interval_gate_passed"] for row in seed_reports)
        reports[target] = {
            "status": "evaluated",
            "features": list(features),
            "eicu_rows": int(len(eicu_rows)),
            "mimic_rows": int(len(mimic_rows)),
            "mimic_subjects": int(mimic_rows["subject_id"].nunique()),
            "patient_seed_reports": seed_reports,
            "point_promoted": point_passed,
            "interval_promoted": interval_passed,
        }
        if point_passed:
            radius, coverage = final_payload
            heads[target] = {
                "model": model,
                "feature_columns": list(features),
                "unit": VARIABLE_UNITS[target],
                "lower_bound": VARIABLE_BOUNDS.get(target, (None, None))[0],
                "upper_bound": VARIABLE_BOUNDS.get(target, (None, None))[1],
                "interval_radius": radius if interval_passed else None,
                "interval_coverage": coverage if interval_passed else None,
                "point_external_gate": "7-seed MIMIC patient + care-unit",
                "minimum_observed_features": 4,
            }
    model_dir = args.bundle.parent
    bundle = {
        "schema": "whole_body_jepa.cross_target_nowcast.v1",
        "training_database": "eICU",
        "external_validation_database": "MIMIC-IV",
        "measurement_process_mode": "measurement_pure",
        "release_binding": {
            name: _sha256(model_dir / name)
            for name in (
                "whole_body_joint_jepa.pt",
                "runtime_registry.json",
                "FEATURE_CONTRACT.json",
            )
        },
        "targets": sorted(heads),
        "heads": heads,
    }
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.bundle)
    args.report.write_text(json.dumps({
        "schema": "whole_body_jepa.cross_target_nowcast_audit.v1",
        "training_database": "eICU",
        "external_validation_database": "MIMIC-IV",
        "variables_evaluated": list(variables),
        "point_promoted_targets": sorted(heads),
        "reports": reports,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"evaluated": len(variables), "promoted": sorted(heads)}, indent=2))


if __name__ == "__main__":
    main()
