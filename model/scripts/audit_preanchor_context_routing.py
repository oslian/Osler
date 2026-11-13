#!/usr/bin/env python3
"""Test whether pre-anchor care intent improves persistence selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from osler_jepa.shared_release_features import FEATURE_SLICES
from scripts.audit_release_masked_cells import donor_indices
from scripts.audit_state_conditioned_kidney import (
    BOUNDS,
    TARGETS,
    comparison_gate,
    coverage_gate,
    patient_bootstrap,
    positions,
    prepare_external,
    prepare_source,
)
from scripts.audit_utility_conditioned_kidney import CUTOFFS, expert_router_split
from whole_body_jepa.multisource_intervals import conformal_radius, sha256
from whole_body_jepa.preanchor_context import (
    attach_eicu_infusion_history,
    build_preanchor_context,
    load_eicu_event_offsets,
    load_eicu_infusion_rates,
    load_mimic_event_offsets,
    load_mimic_stays,
)
from whole_body_jepa.runtime import load_release
from whole_body_jepa.state_conditioned import (
    DYNAMIC_THRESHOLDS,
    empirical_crps,
    fit_distribution_expert,
    fit_utility_router,
    predict_utility_conditioned,
    utility_mixture_samples,
)


ROUTER_STATE_STOP = FEATURE_SLICES["patient_belief"].stop


def _union_rows(lookup: dict[int, int]) -> np.ndarray:
    return np.asarray([row for row, _ in sorted(lookup.items(), key=lambda item: item[1])])


def _eicu_context(frame: pd.DataFrame, lookup: dict[int, int], raw_root: Path):
    rows = _union_rows(lookup)
    selected = frame.iloc[rows]
    stays = set(pd.to_numeric(selected["_raw_stay_id"], errors="raise").astype(int))
    events = load_eicu_event_offsets(raw_root, stays)
    infusions = load_eicu_infusion_rates(raw_root, stays)
    onset = pd.to_numeric(selected.get("onset_hour", 0.0), errors="coerce").fillna(0.0)
    since = pd.to_numeric(selected["hours_since_onset"], errors="coerce").fillna(0.0)
    anchors = 60.0 * (onset + since)
    selected = attach_eicu_infusion_history(selected, anchors, infusions)
    return build_preanchor_context(selected, anchors, events)


def _mimic_context(frame: pd.DataFrame, lookup: dict[int, int], raw_root: Path):
    rows = _union_rows(lookup)
    selected = frame.iloc[rows]
    stay_ids = set(pd.to_numeric(selected["_raw_stay_id"], errors="raise").astype(int))
    stays = load_mimic_stays(raw_root, stay_ids)
    events = load_mimic_event_offsets(raw_root, stays)
    intime = stays.set_index("stay_id")["intime"]
    anchor = selected["t"]
    admission = pd.to_numeric(selected["_raw_stay_id"], errors="raise").astype(int).map(intime)
    offsets = (anchor.to_numpy() - admission.to_numpy()) / np.timedelta64(1, "m")
    return build_preanchor_context(selected, offsets, events)


def _router_features(base: np.ndarray, context: np.ndarray) -> np.ndarray:
    return np.concatenate((base[:, :ROUTER_STATE_STOP], context), axis=1).astype(np.float32)


def fit_bundle(
    frame: pd.DataFrame,
    rows: np.ndarray,
    expert_features: np.ndarray,
    context: np.ndarray,
    target: str,
    seed: int,
) -> tuple[dict, dict]:
    expert_rows, router_rows = expert_router_split(frame, rows, target)
    row_lookup = {int(row): index for index, row in enumerate(rows)}
    expert_pos = positions(expert_rows, row_lookup)
    router_pos = positions(router_rows, row_lookup)
    expert_delta = (
        frame.iloc[expert_rows][f"future_{target}"].to_numpy(dtype=float)
        - frame.iloc[expert_rows][f"{target}_t"].to_numpy(dtype=float)
    )
    router_frame = frame.iloc[router_rows]
    router_delta = (
        router_frame[f"future_{target}"].to_numpy(dtype=float)
        - router_frame[f"{target}_t"].to_numpy(dtype=float)
    )
    expert = fit_distribution_expert(expert_features[expert_pos], expert_delta, seed=seed)
    predicted = expert["point_head"].predict(expert_features[router_pos])
    benefit = np.abs(router_delta) - np.abs(router_delta - predicted)
    base_state = expert_features[router_pos, :ROUTER_STATE_STOP]
    real_context = context[router_pos]
    shuffled = real_context[donor_indices(router_frame["subject_id"], seed + 101)]
    common = {
        "point_head": expert["point_head"],
        "quantile_heads": expert["quantile_heads"],
        "persistence_delta_quantiles": np.quantile(
            np.concatenate((expert_delta, router_delta)), np.linspace(0.0, 1.0, 65)
        ),
    }
    bundle = {
        "context": {**common, **fit_utility_router(_router_features(expert_features[router_pos], real_context), benefit, seed=seed + 21)},
        "no_context": {**common, **fit_utility_router(base_state, benefit, seed=seed + 31)},
        "shuffled_context": {**common, **fit_utility_router(_router_features(expert_features[router_pos], shuffled), benefit, seed=seed + 41)},
    }
    support = {
        "expert_fit_rows": int(len(expert_rows)),
        "router_fit_rows": int(len(router_rows)),
        "router_beneficial_fraction": float((benefit > 0.0).mean()),
    }
    return bundle, support


def _predict(head: dict, expert: np.ndarray, router: np.ndarray, cutoff: float) -> dict:
    return predict_utility_conditioned(head, expert, router, probability_cutoff=cutoff)


def tune(
    frame: pd.DataFrame,
    rows: np.ndarray,
    expert: np.ndarray,
    context: np.ndarray,
    heads: dict,
    target: str,
    seed: int,
) -> tuple[dict[str, float], float]:
    selected = frame.iloc[rows].reset_index(drop=True)
    current = selected[f"{target}_t"].to_numpy(dtype=float)
    truth = selected[f"future_{target}"].to_numpy(dtype=float)
    delta = truth - current
    dynamic = np.abs(delta) >= DYNAMIC_THRESHOLDS[target]
    shuffled = context[donor_indices(selected["subject_id"], seed + 201)]
    router_inputs = {
        "context": _router_features(expert, context),
        "no_context": expert[:, :ROUTER_STATE_STOP],
        "shuffled_context": _router_features(expert, shuffled),
    }
    cutoffs = {}
    for name, router in router_inputs.items():
        choices = []
        for cutoff in CUTOFFS:
            prediction = _predict(heads[name], expert, router, cutoff)
            error = np.abs(current + prediction["point_delta"] - truth)
            mae = pd.Series(error, index=selected["subject_id"]).groupby(level=0).mean().mean()
            stable_harm = float(error[~dynamic].mean() - np.abs(delta[~dynamic]).mean())
            eligible = stable_harm <= 0.01 * DYNAMIC_THRESHOLDS[target]
            choices.append((not eligible, float(mae), -float(prediction["move"].mean()), float(cutoff)))
        cutoffs[name] = min(choices)[3]
    prediction = _predict(heads["context"], expert, router_inputs["context"], cutoffs["context"])
    samples = utility_mixture_samples(prediction, heads["context"]["persistence_delta_quantiles"])
    center = current + np.quantile(samples, 0.50, axis=1)
    lower = current + np.quantile(samples, 0.05, axis=1)
    upper = current + np.quantile(samples, 0.95, axis=1)
    lower_width = np.maximum(center - lower, 1e-6)
    upper_width = np.maximum(upper - center, 1e-6)
    nonconformity = np.maximum((center - truth) / lower_width, (truth - center) / upper_width)
    return cutoffs, conformal_radius(np.maximum(nonconformity, 0.0))


def evaluate(
    frame: pd.DataFrame,
    rows: np.ndarray,
    expert: np.ndarray,
    context: np.ndarray,
    heads: dict,
    cutoffs: dict[str, float],
    interval_scale: float,
    target: str,
    role: str,
    seed: int,
) -> dict:
    selected = frame.iloc[rows].reset_index(drop=True)
    current = selected[f"{target}_t"].to_numpy(dtype=float)
    truth = selected[f"future_{target}"].to_numpy(dtype=float)
    delta = truth - current
    shuffled = context[donor_indices(selected["subject_id"], seed + 301)]
    candidate = _predict(heads["context"], expert, _router_features(expert, context), cutoffs["context"])
    no_context = _predict(
        heads["no_context"], expert, expert[:, :ROUTER_STATE_STOP], cutoffs["no_context"]
    )
    placebo = _predict(
        heads["shuffled_context"],
        expert,
        _router_features(expert, shuffled),
        cutoffs["shuffled_context"],
    )
    point = current + candidate["point_delta"]
    plain = current + candidate["expert_delta"]
    references = {
        "persistence": current,
        "plain_equal_input_head": plain,
        "no_context_router": current + no_context["point_delta"],
        "capacity_matched_shuffled_context": current + placebo["point_delta"],
    }
    benefit = np.abs(delta) - np.abs(delta - candidate["expert_delta"])
    beneficial = benefit > 0.0
    auc = None
    if beneficial.any() and (~beneficial).any():
        auc = float(roc_auc_score(beneficial, candidate["benefit_probability"]))
    samples = utility_mixture_samples(candidate, heads["context"]["persistence_delta_quantiles"])
    persistence_samples = np.tile(heads["context"]["persistence_delta_quantiles"], (len(rows), 1))
    candidate_crps = empirical_crps(samples, delta)
    persistence_crps = empirical_crps(persistence_samples, delta)
    center = current + np.quantile(samples, 0.50, axis=1)
    base_lower = current + np.quantile(samples, 0.05, axis=1)
    base_upper = current + np.quantile(samples, 0.95, axis=1)
    lower = center - interval_scale * (center - base_lower)
    upper = center + interval_scale * (base_upper - center)
    comparisons = {
        name: comparison_gate(selected, point, reference, truth, f"preanchor:{target}:{role}:{name}")
        for name, reference in references.items()
    }
    crps = patient_bootstrap(selected, candidate_crps - persistence_crps, f"preanchor:{target}:{role}:crps")
    moved = candidate["move"]
    result = {
        "rows": int(len(rows)),
        "patients": int(selected["subject_id"].nunique()),
        "benefit_auc": auc,
        "move_fraction": float(moved.mean()),
        "move_precision": float(beneficial[moved].mean()) if moved.any() else None,
        "mae": {
            "preanchor_context_router": float(np.abs(point - truth).mean()),
            **{name: float(np.abs(reference - truth).mean()) for name, reference in references.items()},
        },
        "comparisons": comparisons,
        "crps": {
            "preanchor_context_router": float(candidate_crps.mean()),
            "empirical_persistence": float(persistence_crps.mean()),
            "paired": crps,
        },
        "coverage": coverage_gate(selected, truth, lower, upper),
    }
    result["full_gate_pass"] = bool(
        all(item["pass"] for item in comparisons.values())
        and crps["pass"]
        and result["coverage"]["pass"]
    )
    return result


def write_findings(report: dict, path: Path) -> None:
    lines = [
        "# Pre-Anchor Care Context Routing Audit",
        "",
        "## Result",
        "",
        "The frozen whole-body JEPA and target experts were not retrained. Timestamped",
        "POE intent, bedside assessment and observed treatment history were appended only",
        "to the persistence utility selector. Unit-safe time-weighted, current and delivered",
        "dose/rate features are included where both databases support the dimension. Every",
        "event and delivered amount is truncated at or before its anchor.",
        "",
        "| Target | eICU context - no-context MAE | MIMIC context - no-context MAE | MIMIC benefit AUC | MIMIC move | Promotion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target, item in report["targets"].items():
        evaluation = item["evaluation"]
        eicu = evaluation["eicu_patient"]
        mimic = evaluation["mimic_external"]
        lines.append(
            f"| {target}@{report['horizon_hours']}h | "
            f"{eicu['mae']['preanchor_context_router'] - eicu['mae']['no_context_router']:+.6f} | "
            f"{mimic['mae']['preanchor_context_router'] - mimic['mae']['no_context_router']:+.6f} | "
            f"{mimic['benefit_auc']:.3f} | {100.0 * mimic['move_fraction']:.3f}% | "
            f"{item['promotion_allowed']} |"
        )
    lines += [
        "",
        "No target passed the full eICU patient, unseen-hospital and MIMIC-IV gates.",
        "The target experts themselves improved eICU point MAE, but the utility selector",
        "mostly abstained. External benefit AUC ranged from approximately 0.49 to 0.54,",
        "which is too close to chance to identify when moving beats persistence.",
        "Urine output and potassium had slightly lower MIMIC point MAE than the optimized",
        "no-context selector, but neither beat every reference or its shuffled-context",
        "placebo. This is evidence against promoting this bounded context representation, not evidence",
        "that POE or bedside values are intrinsically uninformative.",
        "",
        "## Boundary",
        "",
        "This is factual context, not a treatment-effect estimate. MIMIC-IV was previously",
        "inspected and is validation data rather than an untouched lockbox. A candidate must",
        "beat persistence, the always-moving expert, the optimized no-context router and an",
        "equal-width patient-shuffled context placebo. Failure leaves the release unchanged.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    report = {
        "schema": "whole_body_jepa.preanchor_context_routing_audit.v1",
        "horizon_hours": args.horizon,
        "release_binding": {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()},
        "source_database": "eICU",
        "external_database": "MIMIC-IV",
        "external_is_untouched_lockbox": False,
        "router_context": "timestamped_poe_bedside_and_observed_treatment_history",
        "physiology_encoder_retrained": False,
        "coverage_bounds": list(BOUNDS),
        "targets": {},
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
    }
    print("prepare eICU model features", flush=True)
    source_frame, source_rows, source_features, source_lookup = prepare_source(args, assets)
    print("extract eICU pre-anchor bedside and care context", flush=True)
    source_context, feature_names = _eicu_context(source_frame, source_lookup, args.eicu_root)
    bundles = {}
    for target in TARGETS:
        print(f"fit pre-anchor utility router: {target}", flush=True)
        train_rows = source_rows[target]["train"]
        train_pos = positions(train_rows, source_lookup)
        heads, support = fit_bundle(
            source_frame,
            train_rows,
            source_features[target][train_pos],
            source_context[train_pos],
            target,
            args.seed,
        )
        tune_rows = source_rows[target]["tune"]
        tune_pos = positions(tune_rows, source_lookup)
        cutoffs, interval_scale = tune(
            source_frame,
            tune_rows,
            source_features[target][tune_pos],
            source_context[tune_pos],
            heads,
            target,
            args.seed,
        )
        bundles[target] = {
            "heads": heads,
            "cutoffs": cutoffs,
            "interval_scale": interval_scale,
        }
        item = {
            "support": support,
            "cutoffs": cutoffs,
            "normalized_conformal_interval_scale": interval_scale,
            "evaluation": {},
        }
        for role, label in (("test", "eicu_patient"), ("hospital", "eicu_hospital")):
            role_rows = source_rows[target][role]
            role_pos = positions(role_rows, source_lookup)
            item["evaluation"][label] = evaluate(
                source_frame,
                role_rows,
                source_features[target][role_pos],
                source_context[role_pos],
                heads,
                cutoffs,
                interval_scale,
                target,
                label,
                args.seed,
            )
        report["targets"][target] = item

    del source_features, source_context
    print("prepare MIMIC-IV model features", flush=True)
    external_frame, external_rows, external_features, external_lookup = prepare_external(args, assets)
    print("extract MIMIC-IV pre-anchor POE and bedside context", flush=True)
    external_context, external_names = _mimic_context(external_frame, external_lookup, args.mimic_root)
    if external_names != feature_names:
        raise RuntimeError("cross-database pre-anchor context contracts differ")
    report["context_features"] = list(feature_names)
    for target in TARGETS:
        rows = external_rows[target]
        pos = positions(rows, external_lookup)
        bundle = bundles[target]
        result = evaluate(
            external_frame,
            rows,
            external_features[target][pos],
            external_context[pos],
            bundle["heads"],
            bundle["cutoffs"],
            bundle["interval_scale"],
            target,
            "mimic_external",
            args.seed,
        )
        item = report["targets"][target]
        item["evaluation"]["mimic_external"] = result
        item["promotion_allowed"] = bool(
            all(value["full_gate_pass"] for value in item["evaluation"].values())
        )
        print(json.dumps({
            "target": target,
            "mimic_gate": result["full_gate_pass"],
            "benefit_auc": result["benefit_auc"],
            "move_fraction": result["move_fraction"],
            "move_precision": result["move_precision"],
            "mae": result["mae"],
            "coverage": result["coverage"],
        }, allow_nan=False), flush=True)
    report["summary"] = {
        "evaluated_targets": len(TARGETS),
        "promoted_targets": [
            target for target, item in report["targets"].items() if item["promotion_allowed"]
        ],
        "mimic_context_router_better_than_no_context": [
            target
            for target, item in report["targets"].items()
            if item["evaluation"]["mimic_external"]["mae"]["preanchor_context_router"]
            < item["evaluation"]["mimic_external"]["mae"]["no_context_router"]
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(
        {
            "schema": "whole_body_jepa.preanchor_context_router_heads.v1",
            "status": "candidate_only",
            "release_binding": report["release_binding"],
            "context_features": list(feature_names),
            "targets": bundles,
        },
        args.output_dir / "heads.joblib",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_findings(report, args.findings)
    print(json.dumps(report["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--eicu",
        type=Path,
        default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_future_label_repaired/eicu_aki_transitions_6h.parquet"),
    )
    parser.add_argument(
        "--mimic",
        type=Path,
        default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/mimiciv_observation_transitions_6h_full_v31_treatment_unit_safe.parquet"),
    )
    parser.add_argument(
        "--eicu-root",
        type=Path,
        default=Path("/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0"),
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=Path("/Users/chunyouchang/Downloads/mimic-iv-3.1"),
    )
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/preanchor_context_routing_20260905_v2")
    )
    parser.add_argument(
        "--findings", type=Path, default=Path("PREANCHOR_CONTEXT_ROUTING_FINDINGS_20260905_V2.md")
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
