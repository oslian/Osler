"""Full promotion gate for the sparse-aware pure-JEPA candidate.

This is intentionally separate from the one-split diagnostic audit.  Every
target must pass all of the following before it is marked promotion-ready:

* seven patient-held-out stay bootstrap comparisons against persistence;
* hospital-held-out comparison when hospital identifiers are available;
* split-conformal 90% coverage on every patient split.

The report never promotes a checkpoint by itself.  It produces evidence and a
registry; runtime still requires a separately marked validated checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nonlinear_latent_rollout import load_multihorizon_cohort, set_seed
from sparse_aware_pure_jepa import (
    _forward,
    _split_subjects,
    conformal_calibration,
    fit_one,
)


SEEDS = (7, 11, 19, 23, 37, 53, 71)
MODULES = (
    "aki", "acute_neuro", "cardiac_injury", "cardiovascular_instability",
    "coagulopathy_heme", "electrolyte_acid_base", "endocrine_stress",
    "gi_pancreatic_nutrition", "hepatic_failure", "immune_inflammatory",
    "integumentary_skin_wound", "musculoskeletal_rhabdo", "respiratory",
    "sepsis", "toxic_metabolic",
)


def _split_groups(frame, column, seed, fraction=0.25):
    groups = frame[column].dropna().drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    selected = set(groups[: max(1, int(round(len(groups) * fraction)))].tolist())
    test = frame[column].isin(selected).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


def _bootstrap_ci(groups, delta, seed, samples=400):
    unique = np.unique(groups)
    if len(unique) < 5:
        return None
    by_group = np.asarray([delta[groups == group].mean() for group in unique])
    rng = np.random.default_rng(seed)
    draws = [rng.choice(by_group, len(by_group), replace=True).mean() for _ in range(samples)]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _forward_placebo(model, arrays, rows, seed):
    """Capacity-matched placebo using feature-wise row permutations.

    The same trained network and number of parameters are retained, but the
    measurement/history features are disconnected from the labels.  This is
    deliberately stricter than comparing only against a prevalence baseline.
    """
    _, input_matrix, current, _, _, history, horizon = arrays
    rng = np.random.default_rng(seed)
    placebo_input = input_matrix.copy()
    placebo_current = current.copy()
    for column in range(placebo_input.shape[1]):
        placebo_input[:, column] = placebo_input[rng.permutation(len(placebo_input)), column]
    for column in range(placebo_current.shape[1]):
        placebo_current[:, column] = placebo_current[rng.permutation(len(placebo_current)), column]
    output_keys = ["value", "observation_logit", "log_scale", "stable_logit"]
    if getattr(model, "measurement_time_head", False):
        output_keys.append("next_measurement_time_log")
    output = {key: [] for key in output_keys}
    model.eval()
    with np.errstate(all="ignore"):
        import torch
        with torch.no_grad():
            for start in range(0, len(rows), 1024):
                batch = rows[start : start + 1024]
                result = model(
                    torch.from_numpy(placebo_input[history[batch]]).float(),
                    torch.from_numpy(placebo_current[batch]).float(),
                    torch.from_numpy(horizon[batch]).float(),
                )
                for key in output:
                    output[key].append(result[key].numpy())
    return {key: np.concatenate(value, axis=0) for key, value in output.items()}


def _brier_gate(model, arrays, frame, variables, train_rows, test_rows, seed):
    """Audit observation/stability heads against baseline and placebo."""
    import torch

    scaler, _, current, future, future_mask, _, _ = arrays
    candidate = _forward(model, arrays, test_rows)
    placebo = _forward_placebo(model, arrays, test_rows, seed + 7000)
    train_future = future[train_rows]
    train_mask = future_mask[train_rows].astype(bool)
    test_future = future[test_rows]
    test_mask = future_mask[test_rows].astype(bool)
    groups = frame.iloc[test_rows]["subject_id"].to_numpy()
    report = {}
    for index, target in enumerate(variables):
        observed = test_mask[:, index].astype(np.float32)
        train_observed = train_mask[:, index].astype(np.float32)
        if len(observed) < 30 or train_observed.sum() < 30:
            report[target] = {"status": "insufficient_support"}
            continue
        p_candidate = 1.0 / (1.0 + np.exp(-np.clip(candidate["observation_logit"][:, index], -40, 40)))
        p_placebo = 1.0 / (1.0 + np.exp(-np.clip(placebo["observation_logit"][:, index], -40, 40)))
        prevalence = float(train_observed.mean())
        candidate_delta = (p_candidate - observed) ** 2 - (prevalence - observed) ** 2
        placebo_delta = (p_candidate - observed) ** 2 - (p_placebo - observed) ** 2
        obs_ci_baseline = _bootstrap_ci(groups, candidate_delta, seed + index)
        obs_ci_placebo = _bootstrap_ci(groups, placebo_delta, seed + 1000 + index)

        stable_valid_train = train_mask[:, index]
        stable_valid_test = test_mask[:, index]
        stable_report = {"status": "insufficient_support"}
        if stable_valid_train.sum() >= 30 and stable_valid_test.sum() >= 30:
            stable_train = (np.abs(train_future[stable_valid_train, index] - current[train_rows][stable_valid_train, index]) <= 0.25).astype(np.float32)
            stable_test = (np.abs(test_future[stable_valid_test, index] - current[test_rows][stable_valid_test, index]) <= 0.25).astype(np.float32)
            p_stable = 1.0 / (1.0 + np.exp(-np.clip(candidate["stable_logit"][stable_valid_test, index], -40, 40)))
            p_stable_placebo = 1.0 / (1.0 + np.exp(-np.clip(placebo["stable_logit"][stable_valid_test, index], -40, 40)))
            stable_prev = float(stable_train.mean())
            stable_groups = groups[stable_valid_test]
            stable_delta_baseline = (p_stable - stable_test) ** 2 - (stable_prev - stable_test) ** 2
            stable_delta_placebo = (p_stable - stable_test) ** 2 - (p_stable_placebo - stable_test) ** 2
            stable_ci_baseline = _bootstrap_ci(stable_groups, stable_delta_baseline, seed + 2000 + index)
            stable_ci_placebo = _bootstrap_ci(stable_groups, stable_delta_placebo, seed + 3000 + index)
            stable_report = {
                "status": "PASS" if stable_ci_baseline is not None and stable_ci_placebo is not None
                    and stable_ci_baseline[1] < 0.0 and stable_ci_placebo[1] < 0.0 else "REJECT",
                "test_n": int(stable_valid_test.sum()),
                "baseline_ci_delta": stable_ci_baseline,
                "placebo_ci_delta": stable_ci_placebo,
            }
        report[target] = {
            "observation_status": "PASS" if obs_ci_baseline is not None and obs_ci_placebo is not None
                and obs_ci_baseline[1] < 0.0 and obs_ci_placebo[1] < 0.0 else "REJECT",
            "observation_test_n": int(len(observed)),
            "observation_baseline_ci_delta": obs_ci_baseline,
            "observation_placebo_ci_delta": obs_ci_placebo,
            "stability": stable_report,
        }
    return report


def _evaluate_gate(model, arrays, frame, variables, rows, seed, baseline_rows=None):
    scaler, _, current, future, future_mask, _, _ = arrays
    output = _forward(model, arrays, rows)
    truth = future[rows] * scaler.scales + scaler.medians
    current_raw = current[rows] * scaler.scales + scaler.medians
    groups = frame.iloc[rows]["subject_id"].to_numpy()
    report = {}
    for index, target in enumerate(variables):
        valid = future_mask[rows, index].astype(bool)
        if valid.sum() < 30:
            report[target] = {"status": "insufficient_support", "n": int(valid.sum())}
            continue
        candidate = output["value"][valid, index] * scaler.scales[index] + scaler.medians[index]
        actual = truth[valid, index]
        persist = current_raw[valid, index]
        candidate_error = np.abs(candidate - actual)
        persistence_error = np.abs(persist - actual)
        delta = candidate_error - persistence_error
        ci = _bootstrap_ci(groups[valid], delta, seed + index)
        report[target] = {
            "n": int(valid.sum()),
            "candidate_mae": float(candidate_error.mean()),
            "persistence_mae": float(persistence_error.mean()),
            "delta_vs_persistence": float(delta.mean()),
            "bootstrap_95_ci_delta": ci,
            "significant_vs_persistence": bool(ci is not None and ci[1] < 0.0),
        }
    if baseline_rows is not None:
        report["_measurement_process"] = _brier_gate(
            model, arrays, frame, variables, baseline_rows, rows, seed
        )
    return report


def _fit_split(frame, variables, train_rows, seed, epochs, batch_size):
    fit, calibration = _split_subjects(
        frame.iloc[train_rows].reset_index(drop=True), seed + 1001, 0.20
    )
    calibration = train_rows[calibration]
    fit = train_rows[fit]
    model, arrays, losses = fit_one(frame, variables, fit, seed, epochs, batch_size)
    return model, arrays, fit, calibration, losses


def _run_module(data_root, module, horizons, epochs, batch_size, max_stays):
    frame, variables = load_multihorizon_cohort(
        data_root, module, horizons, max_stays_per_horizon=max_stays, history=5, seed=0
    )
    patient_reports = {}
    for seed in SEEDS:
        train, test = _split_subjects(frame, seed)
        model, arrays, _, calibration, losses = _fit_split(
            frame, variables, train, seed, epochs, batch_size
        )
        patient_reports[str(seed)] = {
            "metrics": _evaluate_gate(model, arrays, frame, variables, test, seed, train),
            "conformal": conformal_calibration(model, arrays, variables, calibration, test),
            "loss_history": losses,
        }

    hospital_report = {"status": "not_available"}
    hospital_count = int(frame["hospitalid"].dropna().nunique()) if "hospitalid" in frame else 0
    if hospital_count >= 4:
        train, test = _split_groups(frame, "hospitalid", 991)
        model, arrays, _, calibration, losses = _fit_split(
            frame, variables, train, 991, epochs, batch_size
        )
        hospital_report = {
            "status": "run",
            "hospital_count": hospital_count,
            "metrics": _evaluate_gate(model, arrays, frame, variables, test, 991, train),
            "conformal": conformal_calibration(model, arrays, variables, calibration, test),
            "loss_history": losses,
        }

    aggregate = {}
    for target in variables:
        patient_rows = [
            report["metrics"].get(target, {})
            for report in patient_reports.values()
            if report["metrics"].get(target, {}).get("status") is None
        ]
        patient_ok = bool(
            len(patient_rows) == len(SEEDS)
            and all(row.get("significant_vs_persistence", False) for row in patient_rows)
        )
        hospital_metric = hospital_report.get("metrics", {}).get(target, {})
        hospital_ok = bool(
            hospital_report.get("status") == "run"
            and hospital_metric.get("significant_vs_persistence", False)
        )
        conformal_rows = [
            report["conformal"].get(target, {})
            for report in patient_reports.values()
        ]
        conformal_ok = bool(
            len(conformal_rows) == len(SEEDS)
            and all(row.get("status") == "PASS" for row in conformal_rows)
        )
        process_reports = [
            report["metrics"].get("_measurement_process", {}).get(target, {})
            for report in patient_reports.values()
        ]
        observation_ok = bool(
            len(process_reports) == len(SEEDS)
            and all(row.get("observation_status") == "PASS" for row in process_reports)
            and hospital_report.get("status") == "run"
            and hospital_report.get("metrics", {}).get("_measurement_process", {}).get(target, {}).get("observation_status") == "PASS"
        )
        stability_ok = bool(
            len(process_reports) == len(SEEDS)
            and all(row.get("stability", {}).get("status") == "PASS" for row in process_reports)
            and hospital_report.get("status") == "run"
            and hospital_report.get("metrics", {}).get("_measurement_process", {}).get(target, {}).get("stability", {}).get("status") == "PASS"
        )
        aggregate[target] = {
            "patient_seeds": len(patient_rows),
            "patient_significant_seeds": sum(
                row.get("significant_vs_persistence", False) for row in patient_rows
            ),
            "patient_gate": patient_ok,
            "hospital_gate": hospital_ok,
            "conformal_gate": conformal_ok,
            "observation_process_gate": observation_ok,
            "stability_gate": stability_ok,
            "validated_full_gate": patient_ok and hospital_ok and conformal_ok,
        }

    return {
        "module": module,
        "variables": list(variables),
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "hospital_count": hospital_count,
        "patient_heldout": patient_reports,
        "hospital_heldout": hospital_report,
        "aggregate": aggregate,
        "promotion_status": "candidate_only",
        "causal_claim_allowed": False,
        "clinical_promotion_allowed": False,
    }


def build_registry(report):
    modules = {}
    for module, result in report.items():
        if "aggregate" not in result:
            continue
        module_registry = {}
        for target, metrics in result["aggregate"].items():
            ready = bool(metrics.get("validated_full_gate", False))
            reasons = []
            for name in ("patient_gate", "hospital_gate", "conformal_gate"):
                if not metrics.get(name, False):
                    reasons.append(name)
            module_registry[target] = {
                "can_move": False,
                "promotion_ready": ready,
                "status": "promotion_ready_checkpoint_retrain_required" if ready else "fallback_persistence",
                "rejection_reasons": reasons if reasons else ["checkpoint_retrain_required"],
                "patient_gate": bool(metrics.get("patient_gate", False)),
                "hospital_gate": bool(metrics.get("hospital_gate", False)),
                "conformal_gate": bool(metrics.get("conformal_gate", False)),
            }
        modules[module] = module_registry
    return {
        "policy": {
            "fail_closed": True,
            "unvalidated_target_action": "persistence",
            "causal_claim_allowed": False,
            "clinical_promotion_allowed": False,
            "checkpoint_retrain_required_after_gate": True,
        },
        "modules": modules,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/Users/chunyouchang/Desktop/llm_project/medical_jepa")
    parser.add_argument("--modules", default=",".join(MODULES))
    parser.add_argument("--horizons", default="1,3,6,12,24,48")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-stays", type=int, default=1000)
    parser.add_argument("--output", default="/private/tmp/sparse_aware_full_gate.json")
    parser.add_argument("--registry-output", default="/private/tmp/sparse_aware_gate_registry.json")
    args = parser.parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(",") if value.strip())
    report = {}
    for module in [value.strip() for value in args.modules.split(",") if value.strip()]:
        print(f"running sparse-aware full gate: {module}", flush=True)
        try:
            report[module] = _run_module(
                args.data_root, module, horizons, args.epochs, args.batch_size, args.max_stays
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            report[module] = {"status": "not_available", "error": str(exc)}
    registry = build_registry(report)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.registry_output).write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"saved {args.output}", flush=True)
    print(f"saved {args.registry_output}", flush=True)


if __name__ == "__main__":
    main()
