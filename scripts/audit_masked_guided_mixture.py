"""Evaluate directly supervised 3- and 5-expert assignment on frozen JEPA state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from scripts.audit_masked_generalization import arm_score, dump, make_pack, source_setup
from scripts.audit_masked_robust_readouts import reference_predictions
from scripts.audit_release_masked_cells import donor_indices, read_examples, select_rows
from scripts.masked_generalization_methods import source_roles
from scripts.masked_guided_mixture_readouts import FAMILIES, components, fit_family, gate_diagnostics, predict
from scripts.masked_mixture_readouts import oracle_diagnostic, predict as free_predict
from scripts.train_release_masked_abc import candidate_inputs, paired_screen
from whole_body_jepa.multisource_intervals import SEEDS, sha256, stable_hash
from whole_body_jepa.runtime import load_release


def route_summary(values, component_count):
    weights, locations, scales = [part.numpy() for part in values]
    per_seed = weights.reshape(len(weights), -1, component_count) * (weights.shape[1] / component_count)
    entropy = -(per_seed * np.log(np.maximum(per_seed, 1e-12))).sum(2)
    span = np.ptp(locations.reshape(len(locations), -1, component_count), axis=2)
    width = scales.reshape(len(scales), -1, component_count).mean(2)
    return {"effective_components_per_seed": float(np.exp(entropy).mean()),
            "mean_max_route_probability": float(per_seed.max(2).mean()),
            "location_span_over_scale": float(np.mean(span / width))}


def evaluate(args, assets, hidden, fitted, free_models, frozen_models, base_report, report):
    for db, path, grid in (("eicu_hospital", args.eicu, True),
                           ("mimic", args.mimic_current, False), ("nwicu", args.nwicu_current, False)):
        if db not in args.databases:
            continue
        frame = read_examples(path, assets, "eicu" if grid else db, 3, grid=grid)
        arrays = candidate_inputs(frame, assets, 3)
        role = source_roles(frame)["hospital_test"] if grid else np.ones(len(frame), bool)
        report["evaluation"][db] = {}
        for target in report["support"]:
            rows = select_rows(frame, role & frame[f"{target}_t"].notna() & frame[f"{target}_age_hr"].between(0, 1),
                               args.max_patients, target)
            item = report["evaluation"][db][target] = {"patients": len(rows), "arms": {}}
            if len(rows) < 200 or target not in fitted:
                item["status"] = "insufficient_support"
                continue
            if len(rows) != base_report["evaluation"][db][target]["patients"]:
                raise ValueError("evaluation cohort changed")
            pack = make_pack(frame, arrays, rows, target, assets, hidden[target])
            x, selected = pack["latent"], frame.iloc[rows]
            idx = assets.checkpoint["variables"].index(target)
            median, scale = float(assets.scaler.medians[idx]), float(assets.scaler.scales[idx])
            y = selected[f"{target}_t"].to_numpy()
            predictions = {}
            reference_checks = {}
            for arm, (values, _) in reference_predictions(frozen_models[target], pack, db, target).items():
                predictions[arm] = values * scale + median
                reference_mae = base_report["evaluation"][db][target]["arms"][arm]["mae"]
                actual_mae = float(np.abs(predictions[arm] - y).mean())
                reference_checks[arm] = {
                    "actual_mae": actual_mae,
                    "reference_mae": float(reference_mae),
                    "absolute_difference": float(abs(actual_mae - reference_mae)),
                    "strict_atol_1e_6_pass": bool(
                        np.isclose(actual_mae, reference_mae, rtol=0, atol=1e-6)
                    ),
                }
            item["frozen_reference_checks"] = reference_checks
            item["frozen_reference_reproduced"] = all(
                check["strict_atol_1e_6_pass"] for check in reference_checks.values()
            )
            for arm in ("wide_point", "single_laplace", "mixture_laplace"):
                predictions[arm] = free_predict(free_models[target][arm]["heads"], [x] * len(SEEDS)) * scale + median
            placebo_states = [x[donor_indices(pack["subjects"], stable_hash("guided_eval_placebo", db, target, seed))]
                               for seed in SEEDS]
            for family, count in FAMILIES.items():
                bundle = fitted[target][family]
                normalized = predict(bundle["heads"], [x] * len(SEEDS))
                pred = normalized * scale + median
                placebo = predict(bundle["placebo_heads"], placebo_states) * scale + median
                matched = predict(bundle["matched_placebo_heads"], placebo_states) * scale + median
                result = arm_score(selected, y, pred, placebo, median, f"guided:{db}:{target}:{family}")
                result["vs_matched_step_placebo"] = paired_screen(selected, y, pred, matched, f"guided_match:{db}:{target}:{family}")
                result["matched_step_placebo_mae"] = float(np.abs(matched - y).mean())
                result["diagnostic_pass"] &= result["vs_matched_step_placebo"]["pass"]
                predictions[family] = pred
                item["arms"][family] = result
                values = components(bundle["heads"], [x] * len(SEEDS))
                routing = route_summary(values, count)
                routing.update(gate_diagnostics(bundle["heads"], [x] * len(SEEDS), pack["y"]))
                route_controls = {}
                for name, salt in (("uniform", None), ("shuffle_137", 137), ("shuffle_271", 271), ("shuffle_409", 409)):
                    donors = None if salt is None else [donor_indices(pack["subjects"], stable_hash(
                        "guided_route", db, target, family, salt, seed)) for seed in SEEDS]
                    control = predict(bundle["heads"], [x] * len(SEEDS), "uniform" if salt is None else "shuffled", donors) * scale + median
                    route_controls[name] = {"mae": float(np.abs(control - y).mean()),
                                            "legal_vs_control": paired_screen(selected, y, pred, control,
                                                                              f"guided_route:{db}:{target}:{family}:{name}")}
                oracle = oracle_diagnostic(values, normalized, (y - median) / scale) * scale + median
                routing["controls"] = route_controls
                routing["oracle"] = {"mae": float(np.abs(oracle - y).mean()), "outcome_aware_not_a_model": True}
                item.setdefault("routing", {})[family] = routing
                comparisons = {base: paired_screen(selected, y, pred, predictions[base],
                                                    f"guided_pair:{db}:{target}:{family}:{base}")
                               for base in ("latent_erm32", "latent_erm", "wide_point", "single_laplace", "mixture_laplace")}
                item.setdefault("paired_diagnostics", {})[family] = comparisons
                item.setdefault("incremental_pass", {})[family] = bool(
                    result["diagnostic_pass"] and all(value["pass"] for value in comparisons.values()) and
                    all(value["legal_vs_control"]["pass"] for value in route_controls.values()))
            chosen = report["source_selection"][target]
            item["arms"]["source_selected_guided"] = {**item["arms"][chosen], "chosen_arm": chosen}
            item["status"] = "evaluated"
            print(f"evaluated {db} {target}: " + " ".join(
                f"{family}={item['arms'][family]['mae']:.4g}({item['incremental_pass'][family]})" for family in FAMILIES), flush=True)
            dump(report, args.output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--eicu", type=Path, default=Path("/private/tmp/eicu_hospital_balanced_examples_3h_20260903.parquet"))
    parser.add_argument("--mimic-current", type=Path, default=Path("/private/tmp/mimiciv_observation_transitions_3h_full_v31_treatment_unit_safe.parquet"))
    parser.add_argument("--nwicu-current", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/masked_generalization_20260903"))
    parser.add_argument("--free-mixture", type=Path, default=Path("artifacts/masked_mixture_readouts_20260903"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/masked_guided_mixture_20260904_v2"))
    parser.add_argument("--max-patients", type=int, default=4000)
    parser.add_argument("--resume-evaluation", action="store_true",
                        help="Reuse completed candidate fitting in output-dir and rerun evaluation only.")
    parser.add_argument("--databases", default="eicu_hospital,mimic,nwicu",
                        help="Comma-separated evaluation databases to run or resume.")
    args = parser.parse_args()
    args.databases = tuple(part.strip() for part in args.databases.split(",") if part.strip())
    if not set(args.databases) <= {"eicu_hospital", "mimic", "nwicu"}:
        raise ValueError("unknown evaluation database")
    if args.output_dir.exists() and not args.resume_evaluation:
        raise ValueError("use a new output directory")
    torch.manual_seed(73)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    binding = {path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()}
    base_report = json.loads((args.free_mixture / "report.json").read_text())
    reference_report = json.loads((args.reference / "report.json").read_text())
    if not base_report["complete"] or binding != base_report["release_binding"] or binding != reference_report["release_binding"]:
        raise ValueError("incompatible complete reference")
    inputs = {str(path): sha256(path) for path in (args.eicu, args.mimic_current, args.nwicu_current)}
    expected_inputs = base_report["input_hashes"]
    expected_mimic_hashes = [
        value for path, value in expected_inputs.items()
        if "mimiciv_observation_transitions_3h" in path
    ]
    if len(expected_mimic_hashes) != 1:
        raise ValueError("reference must contain exactly one MIMIC 3h cohort")
    if inputs[str(args.eicu)] != expected_inputs[str(args.eicu)]:
        raise ValueError("source fitting cohort changed")
    if inputs[str(args.nwicu_current)] != expected_inputs[str(args.nwicu_current)]:
        raise ValueError("NWICU cohort changed")
    free_bundle = joblib.load(args.free_mixture / "readouts.joblib")
    frozen_bundle = joblib.load(args.reference / "frozen_probes.joblib")
    fresh_report = {"schema": "whole_body_jepa.masked_guided_mixture.v1", "release_binding": binding,
              "input_hashes": inputs, "reference_input_hashes": expected_inputs,
              "mimic_rebuild_requires_all_frozen_maes": inputs[str(args.mimic_current)] != expected_mimic_hashes[0],
              "free_mixture_report_sha256": sha256(args.free_mixture / "report.json"),
              "free_mixture_weights_sha256": sha256(args.free_mixture / "readouts.joblib"),
              "reference_weights_sha256": sha256(args.reference / "frozen_probes.joblib"),
              "protocol_sha256": sha256(Path("MASKED_GUIDED_MIXTURE_PROTOCOL_20260904.md")),
              "code_hashes": {str(path): sha256(path) for path in (Path(__file__), Path("scripts/masked_guided_mixture_readouts.py"))},
              "support": {}, "training": {}, "source_selection": {}, "evaluation": {}, "complete": False,
              "promotion_allowed": False, "runtime_modified": False, "forecast_unchanged_by_construction": True,
              "untouched_lockbox_claim": False, "causal_claim_allowed": False, "next_measurement_time_modeled": False}
    if args.resume_evaluation:
        report = json.loads((args.output_dir / "report.json").read_text())
        bundle = joblib.load(args.output_dir / "readouts.joblib")
        if report.get("complete") or bundle.get("release_binding") != binding:
            raise ValueError("resume requires an incomplete, release-compatible artifact")
        fitted = bundle["models"]
        report["input_hashes"] = inputs
        if bundle.get("source_selection") != report.get("source_selection"):
            raise ValueError("saved source selection changed")
    else:
        report = fresh_report
        args.output_dir.mkdir(parents=True)
    targets = sorted({cell["target"] for cell in assets.source_selector["cells"].values()} - {"hypoxemia_event"})
    frame, arrays, roles, packs, hidden = source_setup(args, assets, targets, report)
    if report["source_roles"] != base_report["source_roles"]:
        raise ValueError("source roles changed")
    del frame, arrays, roles
    if not args.resume_evaluation:
        fitted = {}
        torch.set_num_threads(1)
        for target, pack in packs.items():
            fitted[target], info = {}, {}
            for family in FAMILIES:
                fitted[target][family], info[family] = fit_family(family, pack["fit"], pack["tune"], target)
                print(f"fit {target} {family}: selected={info[family]['chosen_steps']}", flush=True)
            report["training"][target] = info
            report["source_selection"][target] = min(FAMILIES, key=lambda family: (
                info[family]["curve"][info[family]["chosen_steps"]]["selection_score"], family))
            dump(report, args.output_dir)
        joblib.dump({"release_binding": binding, "models": fitted, "source_selection": report["source_selection"],
                     "selection_scope": "eicu_tuning_patients_only", "status": "candidate_only"},
                    args.output_dir / "readouts.joblib")
        report["readouts_sha256"] = sha256(args.output_dir / "readouts.joblib")
        report["selection_frozen_before_external_evaluation"] = True
        dump(report, args.output_dir)
    del packs
    torch.set_num_threads(4)
    evaluate(args, assets, hidden, fitted, free_bundle["models"], frozen_bundle["models"], base_report, report)
    if binding != {path.name: sha256(path) for path in args.model_dir.iterdir() if path.is_file()}:
        raise ValueError("serving artifacts changed")
    report["evaluation_code_hashes"] = {
        str(path): sha256(path)
        for path in (Path(__file__), Path("scripts/masked_guided_mixture_readouts.py"))
    }
    report["evaluation_resumed"] = bool(args.resume_evaluation)
    report["complete"] = True
    report["strict_external_reference_reproduction"] = {
        db: all(
            item.get("status") != "evaluated" or item.get("frozen_reference_reproduced", False)
            for item in targets.values()
        )
        for db, targets in report["evaluation"].items()
    }
    report["candidate_full_incremental_passes"] = {
        db: sorted(
            f"{target}:{family}"
            for target, item in targets.items()
            for family, passed in item.get("incremental_pass", {}).items()
            if passed
        )
        for db, targets in report["evaluation"].items()
    }
    report["promotion_allowed"] = False
    report["complete"] = all(report["strict_external_reference_reproduction"].values())
    dump(report, args.output_dir)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
