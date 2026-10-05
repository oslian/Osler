"""Matched eICU/MIMIC test of mean-risk versus conservative utility routing."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from scripts.audit_preanchor_context_routing import (
    ROUTER_STATE_STOP, _eicu_context, _mimic_context, _router_features, fit_bundle,
)
from scripts.audit_release_masked_cells import donor_indices
from scripts.audit_state_conditioned_kidney import (
    TARGETS, comparison_gate, coverage_gate, patient_bootstrap, positions,
    prepare_source, prepare_external,
)
from scripts.audit_utility_conditioned_kidney import CUTOFFS, expert_router_split
from whole_body_jepa.expected_error_routing import fit_expected_error_router, expected_advantage
from whole_body_jepa.multisource_intervals import conformal_radius, sha256, stable_hash
from whole_body_jepa.runtime import load_release
from whole_body_jepa.state_conditioned import (
    DYNAMIC_THRESHOLDS, expert_distribution_samples, empirical_crps,
)


METHODS = ("old_q25", "probability_only", "expected_error", "mean_benefit",
           "expected_error_no_context", "expected_error_placebo",
           "mean_benefit_no_context", "mean_benefit_placebo")
MARGINS = (0.0, 0.02, 0.05, 0.1, 0.2)


def assert_disjoint(frame, roles):
    patient_sets = {name: set(frame.iloc[rows].subject_id) for name, rows in roles.items()}
    for name, patients in patient_sets.items():
        for other, other_patients in patient_sets.items():
            if name < other and patients & other_patients:
                raise ValueError(f"patient overlap: {name}/{other}")


def calibrating_split(frame, rows):
    selection = np.array([stable_hash("expected_error_calibration", p) % 2 == 0
                          for p in frame.iloc[rows].subject_id])
    return rows[selection], rows[~selection]


def fit_routers(frame, rows, features, context, target, seed):
    conservative, support = fit_bundle(frame, rows, features, context, target, seed)
    _, router_rows = expert_router_split(frame, rows, target)
    pos = positions(router_rows, {int(row): i for i, row in enumerate(rows)})
    chosen = frame.iloc[router_rows]
    delta = chosen[f"future_{target}"].to_numpy(float) - chosen[f"{target}_t"].to_numpy(float)
    expert = conservative["context"]
    prediction = expert["point_head"].predict(features[pos])
    donors = donor_indices(chosen.subject_id, seed + 101)
    inputs = {
        "full": _router_features(features[pos], context[pos]),
        "no_context": features[pos, :ROUTER_STATE_STOP],
        "placebo": _router_features(features[pos], context[pos][donors]),
    }
    risk = {name: fit_expected_error_router(x, delta, prediction, seed=seed + 51)
            for name, x in inputs.items()}
    return {"conservative": conservative, "risk": risk}, support


def predictions(bundle, features, context, subjects, seed):
    head = bundle["conservative"]["context"]
    expert = head["point_head"].predict(features)
    quantiles = np.column_stack([head["quantile_heads"][str(q)].predict(features)
                                 for q in (0.05, 0.5, 0.95)])
    samples = expert_distribution_samples(quantiles)
    u = (np.arange(samples.shape[1]) + 0.5) / samples.shape[1]
    persistence = np.interp(u, np.linspace(0, 1, 65), head["persistence_delta_quantiles"])
    donors = donor_indices(subjects, seed + 301)
    inputs = {
        "full": _router_features(features, context),
        "no_context": features[:, :ROUTER_STATE_STOP],
        "placebo": _router_features(features, context[donors]),
    }
    probability = head["benefit_classifier"].predict_proba(inputs["full"])[:, 1]
    q25 = head["conservative_benefit_head"].predict(inputs["full"])
    scores = {"old_q25": probability, "probability_only": probability}
    for method, direct in (("expected_error", False), ("mean_benefit", True)):
        for variant in inputs:
            name = method if variant == "full" else f"{method}_{variant}"
            scores[name] = expected_advantage(bundle["risk"][variant], inputs[variant], direct=direct)
    return {"expert": expert, "expert_samples": samples, "persistence_samples": persistence,
            "scores": scores, "q25": q25}


def movement(pred, method, threshold):
    move = pred["scores"][method] > threshold
    if method == "old_q25":
        move &= pred["q25"] > 0
    return move


def select_thresholds(pred, delta, target):
    choices, tune_mae = {}, {}
    for method in METHODS:
        grid = CUTOFFS if method in ("old_q25", "probability_only") else tuple(
            x * DYNAMIC_THRESHOLDS[target] for x in MARGINS)
        options = []
        for threshold in grid:
            move = movement(pred, method, threshold)
            loss = np.abs(delta - np.where(move, pred["expert"], 0))
            options.append((float(loss.mean()), float(move.mean()), float(threshold)))
        score, _, choices[method] = min(options)
        tune_mae[method] = score
    primary = min(("expected_error", "mean_benefit"), key=lambda name: tune_mae[name])
    return choices, tune_mae, primary


def interval_parts(pred, move):
    samples = np.where(move[:, None], pred["expert_samples"], pred["persistence_samples"][None, :])
    center = np.quantile(samples, .5, axis=1)
    left = np.maximum(center - np.quantile(samples, .05, axis=1), 1e-6)
    right = np.maximum(np.quantile(samples, .95, axis=1) - center, 1e-6)
    return samples, center, left, right


def calibrate(pred, delta, thresholds):
    scales = {}
    for method, threshold in thresholds.items():
        _, center, left, right = interval_parts(pred, movement(pred, method, threshold))
        scores = np.maximum.reduce(((center - delta) / left, (delta - center) / right,
                                    np.zeros(len(delta))))
        scales[method] = conformal_radius(scores)
    return scales


def evaluate(frame, rows, pred, thresholds, scales, target, role, primary):
    selected = frame.iloc[rows].reset_index(drop=True)
    current = selected[f"{target}_t"].to_numpy(float)
    truth = selected[f"future_{target}"].to_numpy(float)
    delta = truth - current
    points = {name: current + np.where(movement(pred, name, thresholds[name]), pred["expert"], 0)
              for name in METHODS}
    points.update(persistence=current, always_expert=current + pred["expert"])
    beneficial = np.abs(delta - pred["expert"]) < np.abs(delta)
    oracle = current + np.where(beneficial, pred["expert"], 0)
    dynamic = np.abs(delta) >= DYNAMIC_THRESHOLDS[target]
    result = {"rows": len(rows), "patients": int(selected.subject_id.nunique()),
              "oracle_mae_diagnostic_only": float(np.abs(truth - oracle).mean()),
              "reference_mae": {name: float(np.abs(truth - points[name]).mean())
                                for name in ("persistence", "always_expert")}, "methods": {}}
    persistence_crps = empirical_crps(np.tile(pred["persistence_samples"], (len(rows), 1)), delta)
    for method in METHODS:
        point = points[method]
        move = movement(pred, method, thresholds[method])
        samples, center, left, right = interval_parts(pred, move)
        lower = current + center - scales[method] * left
        upper = current + center + scales[method] * right
        hit = (truth >= lower) & (truth <= upper)
        item = {
            "mae": float(np.abs(point - truth).mean()), "move_fraction": float(move.mean()),
            "move_precision": float(beneficial[move].mean()) if move.any() else None,
            "benefit_auc": float(roc_auc_score(beneficial, pred["scores"][method]))
                if len(np.unique(beneficial)) == 2 else None,
            "stable_mae": float(np.abs(point[~dynamic] - truth[~dynamic]).mean()) if (~dynamic).any() else None,
            "dynamic_mae": float(np.abs(point[dynamic] - truth[dynamic]).mean()) if dynamic.any() else None,
            "coverage": coverage_gate(selected, truth, lower, upper),
            "coverage_by_careunit": {str(unit): {"n": int(mask.sum()), "coverage": float(hit[mask].mean())}
                for unit in selected.careunit.unique() for mask in [selected.careunit.eq(unit).to_numpy()]},
            "crps": float(empirical_crps(samples, delta).mean()),
        }
        if method in (primary, "probability_only"):
            references = ["persistence", "always_expert", "old_q25"]
            if method == primary:
                references += [f"{method}_no_context", f"{method}_placebo"]
            item["comparisons"] = {name: comparison_gate(selected, point, points[name], truth,
                f"expected:{target}:{role}:{method}:{name}") for name in references}
            item["crps_gate"] = patient_bootstrap(selected, empirical_crps(samples, delta) - persistence_crps,
                                                   f"expected:{target}:{role}:{method}:crps")
            item["full_gate_pass"] = bool(all(v["pass"] for v in item["comparisons"].values())
                                          and item["coverage"]["pass"] and item["crps_gate"]["pass"])
        result["methods"][method] = item
    return result


def findings(report, path):
    lines = ["# Expected-error routing results", "", "Development decisions frozen on eICU only.",
             "MIMIC is an inspected transfer-validation set, not a new lockbox.", "",
             "| Target 6h | Selected rule | Persistence MAE | Always expert | Old q25 | New MAE | Move % | All-role gate |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for target, item in report["targets"].items():
        ext = item["evaluation"]["mimic"]
        new = ext["methods"][item["primary"]]
        lines.append(f"| {target} | {item['primary']} | {ext['reference_mae']['persistence']:.5f} | "
                     f"{ext['reference_mae']['always_expert']:.5f} | {ext['methods']['old_q25']['mae']:.5f} | "
                     f"{new['mae']:.5f} | {100*new['move_fraction']:.2f} | {item['all_role_gate_pass']} |")
    lines += ["", "## Interpretation limits", "",
              "Seven seeds resample evaluation patients; this is not seven independent training runs.",
              "The time check is late-in-stay, not a calendar-time holdout. Care-unit gate is unit-balanced,",
              "not proof that every individual care unit passes. Per-unit coverage is reported separately.",
              "The current release router is not among these fixed two-expert comparators; runtime promotion",
              "also needs that comparison and a new lockbox. All artifacts remain candidate-only.",
              "The shared encoder is frozen and its prior training exposure is not audited here.",
              "The snapshot/bedside availability fixes apply to every arm. Historical dose fields inherited",
              "from older transition extractors still have incomplete event-availability provenance.",
              "Bedside values are still summarized as categories; this experiment changes routing objective,",
              "and does not establish that available clinical information is exhausted.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    assets = load_release(args.model_dir)
    report = {"schema": "expected_error_routing.v1", "targets": {},
              "protocol_sha256": sha256(Path("EXPECTED_ERROR_ROUTING_PROTOCOL_20260905.md")),
              "release_binding": {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()},
              "promotion_allowed": False, "runtime_modified": False, "causal_claim_allowed": False}
    print("prepare eICU representation", flush=True)
    frame, rows, features, lookup = prepare_source(args, assets)
    for role_rows in rows.values():
        selection, calibration = calibrating_split(frame, role_rows["tune"])
        assert_disjoint(frame, {"train": role_rows["train"], "selection": selection,
                               "calibration": calibration, "test": role_rows["test"],
                               "hospital": role_rows["hospital"]})
    print("extract corrected eICU context", flush=True)
    context, names = _eicu_context(frame, lookup, args.eicu_root)
    report["context_features"] = list(names)
    report["context_support_eicu"] = dict(zip(names, (context != 0).mean(axis=0).tolist()))
    bundles = {}
    for target in TARGETS:
        print(f"fit matched routing objectives: {target}", flush=True)
        role_rows = rows[target]
        selection, calibration = calibrating_split(frame, role_rows["tune"])
        train_pos = positions(role_rows["train"], lookup)
        bundle, support = fit_routers(frame, role_rows["train"], features[target][train_pos],
                                     context[train_pos], target, args.seed)
        def infer(role):
            pos = positions(role, lookup)
            return predictions(bundle, features[target][pos], context[pos], frame.iloc[role].subject_id, args.seed)
        def delta(role):
            f = frame.iloc[role]
            return f[f"future_{target}"].to_numpy(float) - f[f"{target}_t"].to_numpy(float)
        thresholds, tune_mae, primary = select_thresholds(infer(selection), delta(selection), target)
        scales = calibrate(infer(calibration), delta(calibration), thresholds)
        bundles[target] = {"model": bundle, "thresholds": thresholds, "scales": scales, "primary": primary}
        item = {"primary": primary, "support": support, "selection_patients": len(selection),
                "calibration_patients": len(calibration), "selection_mae": tune_mae,
                "thresholds": thresholds, "interval_scales": scales, "evaluation": {}}
        for role in ("test", "hospital"):
            item["evaluation"][role] = evaluate(frame, role_rows[role], infer(role_rows[role]),
                                               thresholds, scales, target, role, primary)
        report["targets"][target] = item
        print(json.dumps({"target": target, "primary": primary, "selection_mae": tune_mae}), flush=True)
    # Persist all choices before reading external outcomes.
    joblib.dump(bundles, args.output_dir / "candidate_heads.joblib")
    (args.output_dir / "development.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    del features, context, frame
    print("prepare MIMIC representation", flush=True)
    frame, rows, features, lookup = prepare_external(args, assets)
    print("extract corrected MIMIC POE/bedside context", flush=True)
    context, external_names = _mimic_context(frame, lookup, args.mimic_root)
    if names != external_names:
        raise ValueError("context feature order mismatch")
    report["context_support_mimic"] = dict(zip(names, (context != 0).mean(axis=0).tolist()))
    for target in TARGETS:
        b = bundles[target]
        pos = positions(rows[target], lookup)
        pred = predictions(b["model"], features[target][pos], context[pos], frame.iloc[rows[target]].subject_id, args.seed)
        result = evaluate(frame, rows[target], pred, b["thresholds"], b["scales"], target, "mimic", b["primary"])
        item = report["targets"][target]
        item["evaluation"]["mimic"] = result
        item["all_role_gate_pass"] = all(e["methods"][b["primary"]]["full_gate_pass"]
                                         for e in item["evaluation"].values())
        print(json.dumps({"target": target, "primary": b["primary"], "refs": result["reference_mae"],
                          "methods": {k: {key: v[key] for key in ("mae", "move_fraction", "benefit_auc")}
                                      for k, v in result["methods"].items()},
                          "all_role_gate_pass": item["all_role_gate_pass"]}), flush=True)
    report["gate_pass_targets"] = [t for t, v in report["targets"].items() if v["all_role_gate_pass"]]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    findings(report, args.output_dir / "FINDINGS.md")
    print("FINISHED " + json.dumps(report["gate_pass_targets"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--eicu", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_future_label_repaired/eicu_aki_transitions_6h.parquet"))
    parser.add_argument("--mimic", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/mimiciv_observation_transitions_6h_full_v31_treatment_unit_safe.parquet"))
    parser.add_argument("--eicu-root", type=Path, default=Path("/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0"))
    parser.add_argument("--mimic-root", type=Path, default=Path("/Users/chunyouchang/Downloads/mimic-iv-3.1"))
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/expected_error_routing_20260905"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
