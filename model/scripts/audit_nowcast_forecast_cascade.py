"""Test whether validated current-state nowcasts improve future JEPA forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from whole_body_jepa.runtime import load_release


TARGETS = ("bicarbonate", "heart_rate")
HORIZON = 6


def _patient_sample(frame: pd.DataFrame, maximum_subjects: int) -> pd.DataFrame:
    ordered = frame.sort_values(["subject_id", "_hours"], kind="stable")
    sampled = ordered.groupby("subject_id", sort=False).first().reset_index()
    if len(sampled) > maximum_subjects:
        sampled = sampled.sort_values("subject_id", kind="stable").iloc[:maximum_subjects]
    return sampled.reset_index(drop=True)


def _time_hours(frame: pd.DataFrame) -> np.ndarray:
    if "hours_since_onset" in frame:
        return pd.to_numeric(frame["hours_since_onset"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    timestamps = pd.to_datetime(frame["t"], errors="coerce")
    first = timestamps.groupby(frame["stay_id"]).transform("min")
    return ((timestamps - first).dt.total_seconds() / 3600.0).fillna(0.0).to_numpy(dtype=np.float32)


def _current_matrix(frame: pd.DataFrame, variables: tuple[str, ...], medians: np.ndarray) -> np.ndarray:
    matrix = np.empty((len(frame), len(variables)), dtype=np.float32)
    for index, variable in enumerate(variables):
        column = f"{variable}_t"
        if column not in frame:
            matrix[:, index] = medians[index]
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
        matrix[:, index] = np.where(np.isfinite(values), values, medians[index])
    return matrix


def _model_forecast(assets, frame: pd.DataFrame, target: str, target_values: np.ndarray, batch_size: int) -> np.ndarray:
    variables = tuple(assets.checkpoint["variables"])
    target_index = variables.index(target)
    physical = _current_matrix(frame, variables, assets.scaler.medians)
    physical[:, target_index] = target_values.astype(np.float32)
    current = ((physical - assets.scaler.medians) / assets.scaler.scales).astype(np.float32)
    n_state = len(variables)
    module_count = len(assets.checkpoint["modules"])
    hours = np.clip(frame["_hours"].to_numpy(dtype=np.float32), 0.0, 168.0)
    onset = (hours / 24.0)[:, None]
    elapsed = (hours / 168.0)[:, None]
    mask = np.ones_like(current, dtype=np.float32)
    ages = np.zeros_like(current, dtype=np.float32)
    presence = np.ones((len(frame), module_count), dtype=np.float32)
    matrix = np.concatenate([current, mask, ages, onset, presence, elapsed], axis=1).astype(np.float32)
    if matrix.shape[1] != int(assets.checkpoint["input_dim"]):
        raise ValueError("cascade input width does not match release checkpoint")
    predictions = np.empty(len(frame), dtype=np.float64)
    history_steps = len(assets.checkpoint["architecture"]["history_lags_hours"])
    assets.model.eval()
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        batch = slice(start, stop)
        history = np.repeat(matrix[batch, None, :], history_steps, axis=1)
        with torch.no_grad():
            output = assets.model(
                torch.from_numpy(history).float(),
                torch.from_numpy(current[batch]).float(),
                torch.from_numpy(mask[batch]).float(),
                torch.from_numpy(ages[batch]).float(),
                torch.from_numpy(presence[batch]).float(),
                torch.full((stop - start,), float(HORIZON), dtype=torch.float32),
            )
        normalized = output["value"][:, target_index].cpu().numpy().astype(np.float64)
        predictions[batch] = normalized * assets.scaler.scales[target_index] + assets.scaler.medians[target_index]
    return predictions


def _bootstrap(values: np.ndarray, seed: int, samples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(values, len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _paired_delta(frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, seed: int, samples: int) -> dict[str, object]:
    truth = frame["truth"].to_numpy(dtype=np.float64)
    row_delta = np.abs(candidate - truth) - np.abs(baseline - truth)
    patient = (
        pd.DataFrame({"subject_id": frame["subject_id"].to_numpy(), "delta": row_delta})
        .groupby("subject_id", sort=False)["delta"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    ci = _bootstrap(patient, seed, samples)
    return {"mean": float(patient.mean()), "ci95": ci, "significant": bool(ci[1] < 0.0)}


def _domain_gate(frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, column: str) -> dict[str, object]:
    truth = frame["truth"].to_numpy(dtype=np.float64)
    reports: dict[str, object] = {}
    supported = 0
    improved = 0
    for domain, positions in frame.groupby(column, dropna=False, sort=True).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        subjects = int(frame.iloc[positions]["subject_id"].nunique())
        if subjects < 30:
            continue
        delta = float(np.mean(np.abs(candidate[positions] - truth[positions]) - np.abs(baseline[positions] - truth[positions])))
        passed = delta < 0.0
        reports[str(domain)] = {"subjects": subjects, "delta": delta, "passed": passed}
        supported += subjects
        improved += subjects if passed else 0
    threshold = max(30, int(np.ceil(0.10 * supported)))
    major_regressions = [name for name, row in reports.items() if row["subjects"] >= threshold and not row["passed"]]
    fraction = improved / supported if supported else 0.0
    return {
        "passed": bool(reports) and fraction >= 0.90 and not major_regressions,
        "patient_weighted_pass_fraction": float(fraction),
        "major_regressions": major_regressions,
        "domains": reports,
    }


def _scope(frame: pd.DataFrame, target: str, mode: str, maximum_subjects: int) -> pd.DataFrame:
    current = pd.to_numeric(frame[f"{target}_t"], errors="coerce")
    future = pd.to_numeric(frame[f"{target}_tp{HORIZON}"], errors="coerce")
    age = pd.to_numeric(frame.get(f"{target}_age_hr"), errors="coerce")
    if mode == "artificial_mask":
        selected = current.notna() & age.notna() & age.le(1.0) & future.notna()
    else:
        selected = current.isna() & future.notna()
    rows = frame.loc[selected].copy()
    rows["truth"] = future.loc[selected].to_numpy(dtype=np.float64)
    rows["_hours"] = frame.loc[selected, "_hours"].to_numpy(dtype=np.float32)
    return _patient_sample(rows, maximum_subjects)


def _nowcast(head: dict[str, object], rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = rows[head["feature_columns"]]
    observed = np.isfinite(features.to_numpy(dtype=np.float64)).sum(axis=1)
    usable = observed >= int(head["minimum_observed_features"])
    prediction = np.full(len(rows), np.nan, dtype=np.float64)
    prediction[usable] = head["model"].predict(features.loc[usable])
    if head.get("lower_bound") is not None:
        prediction = np.maximum(prediction, float(head["lower_bound"]))
    if head.get("upper_bound") is not None:
        prediction = np.minimum(prediction, float(head["upper_bound"]))
    return prediction, usable


def _audit_database(assets, source: Path, database: str, target: str, maximum_subjects: int, batch_size: int, samples: int) -> dict[str, object]:
    frame = pd.read_parquet(source)
    frame["_hours"] = _time_hours(frame)
    head = assets.nowcast["heads"][target]
    domain = "hospitalid" if database == "eICU" else "first_careunit"
    output: dict[str, object] = {}
    for mode in ("artificial_mask", "natural_missing"):
        rows = _scope(frame, target, mode, maximum_subjects)
        nowcast, usable = _nowcast(head, rows)
        rows = rows.loc[usable].reset_index(drop=True)
        nowcast = nowcast[usable]
        if rows.empty:
            output[mode] = {"status": "no_support"}
            continue
        missing_values = np.full(len(rows), assets.scaler.medians[tuple(assets.checkpoint["variables"]).index(target)])
        missing_jepa = _model_forecast(assets, rows, target, missing_values, batch_size)
        cascade_jepa = _model_forecast(assets, rows, target, nowcast, batch_size)
        truth = rows["truth"].to_numpy(dtype=np.float64)
        metrics = {
            "rows": int(len(rows)),
            "subjects": int(rows["subject_id"].nunique()),
            "mae": {
                "missing_input_jepa": float(np.mean(np.abs(missing_jepa - truth))),
                "nowcast_persistence": float(np.mean(np.abs(nowcast - truth))),
                "nowcast_fed_jepa": float(np.mean(np.abs(cascade_jepa - truth))),
            },
            "vs_missing_input_jepa": _paired_delta(rows, cascade_jepa, missing_jepa, 71, samples),
            "vs_nowcast_persistence": _paired_delta(rows, cascade_jepa, nowcast, 1071, samples),
            "domain_vs_missing": _domain_gate(rows, cascade_jepa, missing_jepa, domain),
            "domain_vs_nowcast_persistence": _domain_gate(rows, cascade_jepa, nowcast, domain),
        }
        metrics["cascade_gate_passed"] = bool(
            metrics["vs_missing_input_jepa"]["significant"]
            and metrics["vs_nowcast_persistence"]["significant"]
            and metrics["domain_vs_missing"]["passed"]
            and metrics["domain_vs_nowcast_persistence"]["passed"]
        )
        metrics["_raw"] = {
            "rows": rows,
            "missing_input_jepa": missing_jepa,
            "nowcast_persistence": nowcast,
            "nowcast_fed_jepa": cascade_jepa,
        }
        output[mode] = metrics
    return output


def _split_mask(subjects: pd.Series, modulus: int = 3) -> np.ndarray:
    numeric = pd.util.hash_pandas_object(subjects.astype(str), index=False).to_numpy(dtype=np.uint64)
    return (numeric % np.uint64(modulus)) != 0


def _select_residual_shrinkage(raw: dict[str, object]) -> tuple[float, np.ndarray]:
    rows = raw["rows"]
    fit = _split_mask(rows["subject_id"])
    truth = rows["truth"].to_numpy(dtype=np.float64)
    nowcast = raw["nowcast_persistence"]
    correction = raw["nowcast_fed_jepa"] - nowcast
    grid = np.linspace(0.0, 1.0, 21)
    losses = [np.mean(np.abs(nowcast[fit] + weight * correction[fit] - truth[fit])) for weight in grid]
    weight = float(grid[int(np.argmin(losses))])
    return weight, ~fit


def _shrunk_cascade_report(
    eicu_raw: dict[str, object],
    mimic_raw: dict[str, object],
    samples: int,
) -> dict[str, object]:
    weight, eicu_eval = _select_residual_shrinkage(eicu_raw)
    reports: dict[str, object] = {"residual_weight": weight, "selection_database": "eICU"}
    passed = weight > 0.0
    for database, raw, selected, domain in (
        ("eICU", eicu_raw, eicu_eval, "hospitalid"),
        ("MIMIC-IV", mimic_raw, np.ones(len(mimic_raw["rows"]), dtype=bool), "first_careunit"),
    ):
        rows = raw["rows"].loc[selected].reset_index(drop=True)
        missing = raw["missing_input_jepa"][selected]
        nowcast = raw["nowcast_persistence"][selected]
        direct = raw["nowcast_fed_jepa"][selected]
        candidate = nowcast + weight * (direct - nowcast)
        truth = rows["truth"].to_numpy(dtype=np.float64)
        report = {
            "rows": int(len(rows)),
            "subjects": int(rows["subject_id"].nunique()),
            "mae": {
                "missing_input_jepa": float(np.mean(np.abs(missing - truth))),
                "nowcast_persistence": float(np.mean(np.abs(nowcast - truth))),
                "shrunk_cascade": float(np.mean(np.abs(candidate - truth))),
            },
            "vs_missing_input_jepa": _paired_delta(rows, candidate, missing, 2071, samples),
            "vs_nowcast_persistence": _paired_delta(rows, candidate, nowcast, 3071, samples),
            "domain_vs_missing": _domain_gate(rows, candidate, missing, domain),
            "domain_vs_nowcast_persistence": _domain_gate(rows, candidate, nowcast, domain),
        }
        report["gate_passed"] = bool(
            report["vs_missing_input_jepa"]["significant"]
            and report["vs_nowcast_persistence"]["significant"]
            and report["domain_vs_missing"]["passed"]
            and report["domain_vs_nowcast_persistence"]["passed"]
        )
        passed &= report["gate_passed"]
        reports[database] = report
    reports["cascade_promoted"] = bool(passed)
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eicu", type=Path, required=True)
    parser.add_argument("--mimic", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-subjects", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = load_release(args.model_dir)
    targets: dict[str, object] = {}
    for target in TARGETS:
        reports = {
            "eICU": _audit_database(assets, args.eicu, "eICU", target, args.maximum_subjects, args.batch_size, args.bootstrap_samples),
            "MIMIC-IV": _audit_database(assets, args.mimic, "MIMIC-IV", target, args.maximum_subjects, args.batch_size, args.bootstrap_samples),
        }
        promoted = all(
            reports[database][mode].get("cascade_gate_passed", False)
            for database in reports
            for mode in ("artificial_mask", "natural_missing")
        )
        shrunk = _shrunk_cascade_report(
            reports["eICU"]["natural_missing"]["_raw"],
            reports["MIMIC-IV"]["natural_missing"]["_raw"],
            args.bootstrap_samples,
        )
        for database in reports.values():
            for mode in database.values():
                mode.pop("_raw", None)
        targets[target] = {
            "direct_cascade_promoted": promoted,
            "shrunk_cascade": shrunk,
            "cascade_promoted": bool(promoted or shrunk["cascade_promoted"]),
            "databases": reports,
        }
    payload = {
        "schema": "whole_body_jepa.nowcast_forecast_cascade_audit.v1",
        "horizon_hours": HORIZON,
        "targets": targets,
        "promoted_targets": sorted(target for target, row in targets.items() if row["cascade_promoted"]),
        "runtime_policy": "feed nowcast into future JEPA only for promoted targets",
        "causal_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"promoted_targets": payload["promoted_targets"]}, indent=2))


if __name__ == "__main__":
    main()
