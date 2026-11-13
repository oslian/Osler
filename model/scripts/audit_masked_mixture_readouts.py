"""Source-frozen conditional-density readers and routing ablations."""

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
from scripts.masked_mixture_readouts import (
    FAMILIES, ensemble_components, fit_family, oracle_diagnostic, predict, routing_diagnostics,
)
from scripts.masked_robust_readouts import value_strata
from scripts.train_release_masked_abc import candidate_inputs, paired_screen
from whole_body_jepa.multisource_intervals import SEEDS, sha256, stable_hash
from whole_body_jepa.runtime import load_release


def evaluate(args, assets, hidden, fitted, previous, prior, report):
    for db, path, grid in (("eicu_hospital", args.eicu, True),
                           ("mimic", args.mimic_current, False), ("nwicu", args.nwicu_current, False)):
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
            if len(rows) != prior["evaluation"][db][target]["patients"]:
                raise ValueError("reference evaluation cohort changed")
            pack = make_pack(frame, arrays, rows, target, assets, hidden[target])
            x, selected = pack["latent"], frame.iloc[rows]
            idx = assets.checkpoint["variables"].index(target)
            median, scale = float(assets.scaler.medians[idx]), float(assets.scaler.scales[idx])
            y = selected[f"{target}_t"].to_numpy()
            predictions = {}
            for arm, (p, _) in reference_predictions(previous[target], pack, db, target).items():
                predictions[arm] = p * scale + median
                item["arms"][arm] = prior["evaluation"][db][target]["arms"][arm]
                if not np.isclose(np.abs(predictions[arm] - y).mean(), item["arms"][arm]["mae"], rtol=0, atol=1e-6):
                    raise ValueError("reference MAE failed to reproduce")
            placebo_states = [x[donor_indices(pack["subjects"], stable_hash("mixture_eval_placebo", db, target, s))] for s in SEEDS]
            for family in FAMILIES:
                bundle = fitted[target][family]
                pred = predict(bundle["heads"], [x] * len(SEEDS)) * scale + median
                placebo = predict(bundle["placebo_heads"], placebo_states) * scale + median
                matched = predict(bundle["matched_placebo_heads"], placebo_states) * scale + median
                result = arm_score(selected, y, pred, placebo, median, f"mdn:{db}:{target}:{family}")
                result["vs_matched_step_placebo"] = paired_screen(selected, y, pred, matched, f"mdn_match:{db}:{target}:{family}")
                result["matched_step_placebo_mae"] = float(np.abs(matched - y).mean())
                result["diagnostic_pass"] &= result["vs_matched_step_placebo"]["pass"]
                item["arms"][family] = result
                predictions[family] = pred
            heads = fitted[target]["mixture_laplace"]["heads"]
            components = ensemble_components(heads, [x] * len(SEEDS))
            route_stats = routing_diagnostics(components)
            for salt in (137, 271, 409):
                indices = [donor_indices(pack["subjects"], stable_hash("route_only", db, target, salt, s)) for s in SEEDS]
                predictions[f"shuffled_route_{salt}"] = predict(heads, [x] * len(SEEDS), "shuffled", indices) * scale + median
            predictions["uniform_route"] = predict(heads, [x] * len(SEEDS), "uniform") * scale + median
            legal = predictions["mixture_laplace"]
            oracle = oracle_diagnostic(components, (legal - median) / scale, (y - median) / scale) * scale + median
            route_stats["oracle"] = {"mae": float(np.abs(oracle - y).mean()),
                                     "legal_mae": float(np.abs(legal - y).mean()),
                                     "outcome_aware_not_a_model": True}
            route_stats["comparisons"] = {name: {
                "mae": float(np.abs(predictions[name] - y).mean()),
                "legal_vs_control": paired_screen(selected, y, legal, predictions[name], f"routing:{db}:{target}:{name}")}
                for name in ("uniform_route", "shuffled_route_137", "shuffled_route_271", "shuffled_route_409")}
            item["routing"] = route_stats
            item["paired_diagnostics"] = {family: {
                base: paired_screen(selected, y, predictions[family], predictions[base], f"mdn_pair:{db}:{target}:{family}:{base}")
                for base in ("latent_erm32", "latent_erm", "wide_point", "single_laplace") if base != family}
                for family in FAMILIES}
            item["mixture_incremental_pass"] = bool(
                item["arms"]["mixture_laplace"]["diagnostic_pass"] and
                all(v["pass"] for v in item["paired_diagnostics"]["mixture_laplace"].values()) and
                all(v["legal_vs_control"]["pass"] for v in route_stats["comparisons"].values()))
            chosen = report["source_selection"][target]
            item["arms"]["source_selected_new"] = {**item["arms"][chosen], "chosen_arm": chosen}
            item["value_strata"] = value_strata(y, predictions, report["support"][target]["physical_tail_bounds"])
            item["status"] = "evaluated"
            print(f"evaluated {db} {target}: " + " ".join(f"{a}={v['mae']:.4g}({v['diagnostic_pass']})" for a, v in item["arms"].items()), flush=True)
            dump(report, args.output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--eicu", type=Path, default=Path("/private/tmp/eicu_hospital_balanced_examples_3h_20260903.parquet"))
    parser.add_argument("--mimic-current", type=Path, default=Path("/private/tmp/mimiciv_observation_transitions_3h_full_v31_treatment_unit_safe.parquet"))
    parser.add_argument("--nwicu-current", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet"))
    parser.add_argument("--previous", type=Path, default=Path("artifacts/masked_generalization_20260903"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/masked_mixture_readouts_20260903"))
    parser.add_argument("--max-patients", type=int, default=4000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("use a new output directory")
    torch.set_num_threads(4)
    torch.manual_seed(73)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    binding = {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}
    prior = json.loads((args.previous / "report.json").read_text())
    if not prior["complete"] or binding != prior["release_binding"]:
        raise ValueError("incompatible reference")
    inputs = {str(p): sha256(p) for p in (args.eicu, args.mimic_current, args.nwicu_current)}
    if any(prior["input_hashes"].get(p) != h for p, h in inputs.items()):
        raise ValueError("input cohorts changed")
    previous = joblib.load(args.previous / "frozen_probes.joblib")
    if previous["release_binding"] != binding or previous["selection_scope"] != "eicu_tuning_patients_only":
        raise ValueError("incompatible frozen heads")
    report = {"schema": "whole_body_jepa.masked_mixture_readouts.v1", "release_binding": binding,
              "input_hashes": inputs, "previous_report_sha256": sha256(args.previous / "report.json"),
              "previous_weights_sha256": sha256(args.previous / "frozen_probes.joblib"),
              "code_hashes": {str(p): sha256(p) for p in (Path(__file__), Path("scripts/masked_mixture_readouts.py"))},
              "protocol_sha256": sha256(Path("MASKED_MIXTURE_READOUT_PROTOCOL_20260903.md")),
              "support": {}, "training": {}, "source_selection": {}, "evaluation": {},
              "complete": False, "promotion_allowed": False, "runtime_modified": False,
              "forecast_unchanged_by_construction": True, "untouched_lockbox_claim": False,
              "next_measurement_time_modeled": False, "causal_claim_allowed": False}
    args.output_dir.mkdir(parents=True)
    targets = sorted({c["target"] for c in assets.source_selector["cells"].values()} - {"hypoxemia_event"})
    frame, arrays, roles, packs, hidden = source_setup(args, assets, targets, report)
    if report["source_roles"] != prior["source_roles"]:
        raise ValueError("source roles changed")
    del frame, arrays, roles
    fitted = {}
    # Tiny dense readers are faster without intra-op thread scheduling.
    torch.set_num_threads(1)
    for target, pack in packs.items():
        fitted[target], info = {}, {}
        idx = assets.checkpoint["variables"].index(target)
        report["support"][target]["physical_tail_bounds"] = (
            np.quantile(pack["fit"]["y"], [.1, .9]) * assets.scaler.scales[idx] + assets.scaler.medians[idx]).tolist()
        for family in FAMILIES:
            fitted[target][family], info[family] = fit_family(family, pack["fit"], pack["tune"], target)
            print(f"fit {target} {family}: selected={info[family]['chosen_steps']}", flush=True)
        report["training"][target] = info
        report["source_selection"][target] = min(FAMILIES, key=lambda f: (info[f]["curve"][info[f]["chosen_steps"]]["selection_score"], f))
        dump(report, args.output_dir)
    joblib.dump({"release_binding": binding, "models": fitted, "source_selection": report["source_selection"],
                 "selection_scope": "eicu_tuning_patients_only", "status": "candidate_only"}, args.output_dir / "readouts.joblib")
    report["readouts_sha256"] = sha256(args.output_dir / "readouts.joblib")
    report["selection_frozen_before_external_evaluation"] = True
    dump(report, args.output_dir)
    del packs
    torch.set_num_threads(4)
    evaluate(args, assets, hidden, fitted, previous["models"], prior, report)
    if binding != {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}:
        raise ValueError("serving artifacts changed")
    report["complete"] = True
    dump(report, args.output_dir)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
