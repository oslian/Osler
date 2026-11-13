#!/usr/bin/env python3
"""Audit label-independent cached-anchor replay and resolved own-error feedback."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_continual_validated_cell import CELL, eligible_rows, prepare, split_fit_patients
from scripts.audit_event_aware_patient_state import (
    cached_context,
    eicu_treatment_context,
    measurement_matrix,
    mimic_treatment_context,
    paired,
    patient_partition,
    replay_features,
    shuffled_event_features,
    wrong_patient_state,
)
from scripts.audit_measurement_anchor_encoder import load_dataset, selected_rows
from scripts.screen_cross_variable_history_catalog import target_role_masks
from whole_body_jepa.change_evaluation import CHANGE_THRESHOLDS
from whole_body_jepa.multisource_intervals import SEEDS, sha256, subject_digest
from whole_body_jepa.recurrent_event_state import (
    fit_recurrent_signed_residual,
    predict_recurrent,
)
from whole_body_jepa.runtime import load_release
from whole_body_jepa.source_selection import HEAD


PROTOCOL = Path("EVENT_STATE_REPAIR_PROTOCOL_20260910.md")
TARGET = "glucose"


def fit_bundle(frame: pd.DataFrame, features: np.ndarray, seed: int, scale: float) -> dict:
    fit, tune = patient_partition(frame.subject_id, seed)
    truth = frame.truth.to_numpy(float)
    frozen = frame.frozen.to_numpy(float)
    shuffled = shuffled_event_features(frame, features, seed)
    models = {}
    specifications = (
        ("recurrent_event_state", features, False, 0),
        ("own_error_feedback", features, False, 0),
        ("anchor_reset", features, True, 101),
        ("shuffled_event_recurrent", shuffled, False, 211),
    )
    for name, values, reset, offset in specifications:
        standardizer, model, audit = fit_recurrent_signed_residual(
            frame,
            values,
            fit,
            tune,
            truth=truth,
            frozen=frozen,
            residual_scale=scale,
            seed=seed + offset,
            reset_each_anchor=reset,
            own_feedback=name == "own_error_feedback",
        )
        models[name] = {
            "standardizer": standardizer,
            "model": model,
            "audit": audit,
            "reset_each_anchor": reset,
            "own_feedback": name == "own_error_feedback",
        }
    return models


def model_prediction(
    item: dict,
    frame: pd.DataFrame,
    features: np.ndarray,
    frozen: np.ndarray,
    scale: float,
) -> np.ndarray:
    correction = predict_recurrent(
        item["model"],
        item["standardizer"],
        frame,
        features,
        reset_each_anchor=item["reset_each_anchor"],
        feedback_scale=scale if item["own_feedback"] else None,
    )
    return frozen + correction.astype(float) * scale


def patient_equal_mae(frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> float:
    error = pd.Series(
        np.abs(np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)),
        index=frame.subject_id.astype(str).to_numpy(),
    )
    return float(error.groupby(level=0).mean().mean())


def evaluate(
    frame: pd.DataFrame,
    features: np.ndarray,
    bundle: dict,
    *,
    role: str,
    seed: int,
    scale: float,
    bootstrap: int,
) -> dict:
    truth = frame.truth.to_numpy(float)
    current = frame.current.to_numpy(float)
    frozen = frame.frozen.to_numpy(float)
    shuffled = shuffled_event_features(frame, features, seed)
    wrong_features, _ = wrong_patient_state(frame, features, seed)
    predictions = {
        "persistence": current,
        "frozen": frozen,
        "resolved_error_only": frame.resolved_error_only.to_numpy(float),
        "recurrent_event_state": model_prediction(
            bundle["recurrent_event_state"], frame, features, frozen, scale
        ),
        "own_error_feedback": model_prediction(bundle["own_error_feedback"], frame, features, frozen, scale),
        "anchor_reset": model_prediction(
            bundle["anchor_reset"], frame, features, frozen, scale
        ),
        "shuffled_event_recurrent": model_prediction(
            bundle["shuffled_event_recurrent"], frame, shuffled, frozen, scale
        ),
        "wrong_patient_recurrent": model_prediction(
            bundle["recurrent_event_state"], frame, wrong_features, frozen, scale
        ),
    }
    primary = (frame.resolved_history_count.to_numpy(int) > 0) & np.isfinite(truth)
    if not primary.any():
        raise ValueError(f"{role} has no predictions after a resolved outcome")
    work = frame.loc[primary].reset_index(drop=True)
    scoped = {name: values[primary] for name, values in predictions.items()}
    scoped_truth = truth[primary]
    scoped_current = current[primary]
    references = (
        "persistence",
        "frozen",
        "resolved_error_only",
        "anchor_reset",
        "shuffled_event_recurrent",
        "wrong_patient_recurrent",
    )
    comparisons = {
        name: paired(
            work,
            scoped["recurrent_event_state"],
            scoped[name],
            f"recurrent-loop:{role}:{seed}:{name}",
            bootstrap,
        )
        for name in references
    }
    own_comparisons = {name: paired(work, scoped["own_error_feedback"], scoped[name],
                        f"repair-own:{role}:{seed}:{name}", bootstrap)
                       for name in (*references, "recurrent_event_state")}
    actual_delta = scoped_truth - scoped_current
    changed = np.abs(actual_delta) >= float(CHANGE_THRESHOLDS[TARGET])
    treatment = work.treatment_changed.to_numpy(bool)
    subsets = {}
    for name, mask in {
        "changed": changed,
        "stable": ~changed,
        "treatment_transition": treatment,
        "changed_after_treatment_transition": changed & treatment,
    }.items():
        item = {
            "rows": int(mask.sum()),
            "patients": int(work.loc[mask, "subject_id"].nunique()),
        }
        if mask.any():
            item["patient_equal_mae"] = {
                method: patient_equal_mae(
                    work.loc[mask], scoped_truth[mask], value[mask]
                )
                for method, value in scoped.items()
            }
            item["direction_accuracy"] = {
                method: float(
                    np.mean(
                        np.sign(value[mask] - scoped_current[mask])
                        == np.sign(actual_delta[mask])
                    )
                )
                for method, value in scoped.items()
            }
        subsets[name] = item
    return {
        "rows": int(len(work)),
        "patients": int(work.subject_id.nunique()),
        "patient_equal_mae": {
            name: patient_equal_mae(work, scoped_truth, prediction)
            for name, prediction in scoped.items()
        },
        "comparisons": comparisons,
        "own_feedback_comparisons": own_comparisons,
        "subsets": subsets,
    }


def write_findings(report: dict, path: Path) -> None:
    methods = (
        "persistence",
        "frozen",
        "resolved_error_only",
        "anchor_reset",
        "recurrent_event_state",
        "own_error_feedback",
        "shuffled_event_recurrent",
        "wrong_patient_recurrent",
    )

    def mean(role: str, method: str) -> float:
        return float(
            np.mean(
                [
                    seed["evaluation"][role]["patient_equal_mae"][method]
                    for seed in report["seeds"]
                ]
            )
        )

    lines = [
        "# Recurrent Event-State Loop Findings",
        "",
        "## Decision",
        "",
        f"{report['status']}. The active release is unchanged.",
        "",
        "This audit used every available cached anchor for fixed patient roles,",
        "including anchors without scored future labels. It is not a complete",
        "raw-event ledger and available-time/storetime equivalence is unverified.",
        "Own-error feedback only ingests detached candidate errors after resolution.",
        "",
        "| Role | " + " | ".join(methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for role in ("eicu_hospital", "mimic"):
        lines.append(
            "| "
            + role
            + " | "
            + " | ".join(f"{mean(role, method):.3f}" for method in methods)
            + " |"
        )
    lines += [
        "",
        "Values are seven-seed means of patient-equal MAE on forecasts made only",
        "after at least one previous outcome had resolved. anchor_reset has the",
        "same event encoder and signed readout but clears recurrent state at each",
        "anchor. shuffled_event_recurrent preserves capacity while replacing event",
        "context with causal wrong-patient donors.",
        "",
        "## Gate",
        "",
        "The point screen required recurrent event-state to significantly beat frozen JEPA,",
        "persistence, resolved-error-only state, anchor-reset, shuffled-event recurrent",
        "and wrong-patient recurrent controls in all seven source-defined seeds on both",
        "eICU hospital-heldout and MIMIC. Exact CIs and change/treatment subsets are in",
        "the JSON artifact. Full promotion also needs interval and time/care-unit gates.",
        "",
        "This remains factual prediction. causal_claim_allowed is false.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_replay(args: argparse.Namespace) -> tuple:
    """Build the fixed source/patient replay contract for paired diagnostics."""
    assets = load_release(args.model_dir)
    decision = assets.source_selector["cells"][CELL]
    if decision["source_model"] != HEAD or not decision["can_move"]:
        raise ValueError("glucose@6h is not backed by the expected frozen JEPA head")
    eicu, eicu_arrays = load_dataset(args.eicu, "eicu", assets)
    mimic, mimic_arrays = load_dataset(args.mimic, "mimic", assets)
    eicu_eligible = eligible_rows(eicu)
    mimic_eligible = eligible_rows(mimic)
    roles, hospital_metadata = target_role_masks(
        eicu, eicu_eligible, "recurrent-loop:glucose@6h"
    )
    role_patients = {
        role: set(eicu.loc[eicu_eligible & mask, "subject_id"].astype(str))
        for role, mask in roles.items()
    }
    fit_roles = split_fit_patients(role_patients["fit"])
    patient_sets = {
        **fit_roles,
        "development": role_patients["patient_test"],
        "eicu_hospital": role_patients["hospital_test"],
    }
    selected = {}
    prepared = {}
    for role, patients in patient_sets.items():
        rows = selected_rows(
            eicu,
            eicu_eligible if args.scored_only else np.ones(len(eicu), dtype=bool),
            patients,
            anchors=0,
            salt=f"recurrent-loop:{role}",
        )
        selected[role] = rows
        prepared[role] = prepare(eicu, eicu_arrays, rows, assets)
        prepared[role] = replace(prepared[role], truth=np.where(eicu_eligible[rows], prepared[role].truth, np.nan))
    mimic_patients = set(mimic.loc[mimic_eligible, "subject_id"].astype(str))
    selected["mimic"] = selected_rows(
        mimic,
        mimic_eligible if args.scored_only else np.ones(len(mimic), dtype=bool),
        mimic_patients,
        anchors=0,
        salt="recurrent-loop:mimic",
    )
    prepared["mimic"] = prepare(
        mimic, mimic_arrays, selected["mimic"], assets
    )
    prepared["mimic"] = replace(prepared["mimic"], truth=np.where(mimic_eligible[selected["mimic"]], prepared["mimic"].truth, np.nan))
    scale = max(
        float(
            np.nanmedian(
                np.abs(prepared["memory"].truth - prepared["memory"].base)
            )
            / 0.6744897501960817
        ),
        1.0,
    )
    eicu_union = np.unique(
        np.concatenate([selected[name] for name in patient_sets])
    )
    eicu_context, eicu_context_audit = cached_context(
        args.eicu_context_cache,
        eicu_union,
        lambda: eicu_treatment_context(eicu.iloc[eicu_union], args.eicu_root),
    )
    eicu_position = {int(row): index for index, row in enumerate(eicu_union)}
    mimic_context, mimic_context_audit = cached_context(
        args.mimic_context_cache,
        selected["mimic"],
        lambda: mimic_treatment_context(
            mimic.iloc[selected["mimic"]], args.mimic_root
        ),
    )
    replay = {}
    feature = {}
    for role in patient_sets:
        positions = np.asarray(
            [eicu_position[int(row)] for row in selected[role]]
        )
        replay[role], feature[role] = replay_features(
            prepared[role],
            measurement_matrix(prepared[role].frame, assets),
            eicu_context[positions],
            scale,
            reference_center=float(assets.scaler.medians[assets.checkpoint["variables"].index(TARGET)]),
            max_age_hours=float(assets.max_age_hours),
        )
    replay["mimic"], feature["mimic"] = replay_features(
        prepared["mimic"],
        measurement_matrix(prepared["mimic"].frame, assets),
        mimic_context,
        scale,
        reference_center=float(assets.scaler.medians[assets.checkpoint["variables"].index(TARGET)]),
        max_age_hours=float(assets.max_age_hours),
    )
    return replay, feature, scale, hospital_metadata, eicu_context_audit, mimic_context_audit


def run(args: argparse.Namespace) -> None:
    if args.output.exists() or args.findings.exists():
        raise ValueError("refusing to overwrite recurrent event-state outputs")
    replay, feature, scale, hospital_metadata, eicu_context_audit, mimic_context_audit = prepare_replay(args)
    seeds = []
    for seed in SEEDS:
        bundle = fit_bundle(
            replay["development"], feature["development"], int(seed), scale
        )
        evaluation = {
            role: evaluate(
                replay[role],
                feature[role],
                bundle,
                role=role,
                seed=int(seed),
                scale=scale,
                bootstrap=args.bootstrap,
            )
            for role in ("development", "eicu_hospital", "mimic")
        }
        seeds.append(
            {
                "seed": int(seed),
                "fit": {
                    name: item["audit"] for name, item in bundle.items()
                },
                "evaluation": evaluation,
            }
        )
        print(
            json.dumps(
                {
                    "seed": int(seed),
                    "eicu_hospital": evaluation["eicu_hospital"][
                        "patient_equal_mae"
                    ],
                    "mimic": evaluation["mimic"]["patient_equal_mae"],
                },
                allow_nan=False,
            ),
            flush=True,
        )
    references = (
        "persistence",
        "frozen",
        "resolved_error_only",
        "anchor_reset",
        "shuffled_event_recurrent",
        "wrong_patient_recurrent",
    )
    pass_counts = {
        role: {
            reference: int(
                sum(
                    seed["evaluation"][role]["comparisons"][reference]["ci95"]
                    is not None
                    and seed["evaluation"][role]["comparisons"][reference][
                        "ci95"
                    ][1]
                    < 0.0
                    for seed in seeds
                )
            )
            for reference in references
        }
        for role in ("eicu_hospital", "mimic")
    }
    promotion = all(
        pass_counts[role][reference] == len(SEEDS)
        for role in pass_counts
        for reference in references
    )
    report = {
        "schema": "whole_body_jepa.recurrent_event_state_loop.v2",
        "complete": True,
        "status": "candidate_only",
        "point_screen_passed": bool(promotion),
        "promotion_allowed": False,
        "scored_only_stream": bool(args.scored_only),
        "full_raw_event_stream": False,
        "storetime_availability_verified": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "untouched_lockbox": False,
        "cell": CELL,
        "loop": [
            "anchor_known_event_updates_patient_state",
            "state_decodes_signed_residual",
            "forecast_outcome_updates_memory_only_after_resolution",
        ],
        "release_binding": {
            "checkpoint": sha256(
                args.model_dir / "whole_body_joint_jepa.pt"
            ),
            "protocol": sha256(PROTOCOL),
            "script": sha256(Path(__file__)),
            "recurrent_module": sha256(Path("whole_body_jepa/recurrent_event_state.py")),
            "feature_module": sha256(Path("scripts/audit_event_aware_patient_state.py")),
            "treatment_module": sha256(Path("whole_body_jepa/event_treatment.py")),
            "eicu_input": sha256(args.eicu),
            "mimic_input": sha256(args.mimic),
        },
        "hospital_role": hospital_metadata,
        "context_support": {
            "eicu": eicu_context_audit,
            "mimic": mimic_context_audit,
        },
        "support": {
            role: {
                "rows": int(len(frame)),
                "scored_rows": int(np.isfinite(frame.truth).sum()),
                "patients": int(frame.subject_id.nunique()),
                "personalized_rows": int(
                    (frame.resolved_history_count > 0).sum()
                ),
                "subject_digest": subject_digest(frame.subject_id),
            }
            for role, frame in replay.items()
        },
        "residual_scale": scale,
        "seeds": seeds,
        "pass_counts": pass_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_findings(report, args.findings)
    print(
        json.dumps(
            {"status": report["status"], "pass_counts": pass_counts},
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--eicu",
        type=Path,
        default=Path(
            "/private/tmp/eicu_creatinine_measurement_anchor_examples_v2_6h_5000.parquet"
        ),
    )
    parser.add_argument(
        "--mimic",
        type=Path,
        default=Path(
            "/private/tmp/mimiciv_creatinine_measurement_anchor_examples_v2_6h_5000.parquet"
        ),
    )
    parser.add_argument(
        "--eicu-root",
        type=Path,
        default=Path(
            "/Users/chunyouchang/Downloads/eicu-collaborative-research-database-2.0"
        ),
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=Path("/Users/chunyouchang/Downloads/mimic-iv-3.1"),
    )
    parser.add_argument(
        "--eicu-context-cache",
        type=Path,
        default=Path(
            "/private/tmp/event_aware_eicu_glucose_6h_context.npz"
        ),
    )
    parser.add_argument(
        "--mimic-context-cache",
        type=Path,
        default=Path(
            "/private/tmp/event_aware_mimic_glucose_6h_context.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/recurrent_event_state_repaired_20260910.json"
        ),
    )
    parser.add_argument(
        "--findings",
        type=Path,
        default=Path("RECURRENT_EVENT_STATE_REPAIRED_FINDINGS_20260910.md"),
    )
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--scored-only", action="store_true", help="Ablate updates at cached anchors lacking scoring labels")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
