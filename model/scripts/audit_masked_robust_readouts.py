"""Frozen-state robust readout experiments. No serving or external fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from scripts.audit_masked_generalization import arm_score, dump, make_pack, source_setup
from scripts.audit_release_masked_cells import donor_indices, read_examples, select_rows
from scripts.masked_generalization_methods import ensemble_predict, source_roles
from scripts.masked_robust_readouts import FAMILIES, ensemble, fit_family, placebo_ensemble, value_strata
from scripts.train_release_masked_abc import candidate_inputs, paired_screen
from whole_body_jepa.multisource_intervals import SEEDS, sha256, stable_hash
from whole_body_jepa.runtime import load_release


def reference_predictions(bundle, pack, db, target):
    old = bundle["latent_erm"]
    results = {}
    for arm, real_key, placebo_key in (
            ("latent_erm32", "fixed32_heads", "fixed32_placebo_heads"),
            ("latent_erm", "heads", "placebo_heads")):
        controls = [ensemble_predict([model], pack["latent"][donor_indices(
            pack["subjects"], stable_hash("external_placebo", db, target, seed))])
                    for seed, model in zip(SEEDS, old[placebo_key])]
        results[arm] = (ensemble_predict(old[real_key], pack["latent"]), np.mean(controls, axis=0))
    ridge = bundle["latent_ridge"]
    results["latent_ridge"] = (ridge["model"].predict(pack["latent"]), np.mean([
        model.predict(np.asarray(pack["latent"])[donor_indices(
            pack["subjects"], stable_hash("external_placebo", db, target, seed))])
        for seed, model in zip(SEEDS, ridge["placebos"])], axis=0))
    return results


def choose_source_head(info, prior_training):
    erm = prior_training["latent_erm"]
    ridge = prior_training["latent_ridge"]
    scores = {"latent_erm32": erm["curve"]["32"]["selection_score"],
              "latent_erm": erm["curve"][str(erm["chosen_steps"])]["selection_score"],
              "latent_ridge": ridge["curve"][str(ridge["alpha"])]["selection_score"]}
    scores.update({a: v["selection_score"] for a, v in info.items() if v["status"] == "fitted"})
    return {"chosen_arm": min(scores, key=lambda a: (scores[a], a)), "scores": scores}


def evaluate(args, assets, hidden, fitted, previous, prior_report, report):
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
            if prior_report["evaluation"][db][target]["patients"] != len(rows):
                raise ValueError("reference patient cohort changed")
            pack = make_pack(frame, arrays, rows, target, assets, hidden[target])
            x = np.asarray(pack["latent"], dtype=float)
            idx = assets.checkpoint["variables"].index(target)
            median, scale = float(assets.scaler.medians[idx]), float(assets.scaler.scales[idx])
            y = frame.iloc[rows][f"{target}_t"].to_numpy()
            selected = frame.iloc[rows]
            predictions = {}
            for arm, (pred, placebo) in reference_predictions(previous[target], pack, db, target).items():
                predictions[arm] = pred * scale + median
                # Reference scores are the exact previously frozen results, not
                # re-bootstrap draws that could change a borderline decision.
                item["arms"][arm] = prior_report["evaluation"][db][target]["arms"][arm]
                if not np.isclose(np.abs(predictions[arm] - y).mean(), item["arms"][arm]["mae"], rtol=0, atol=1e-6):
                    raise ValueError("reference predictions do not reproduce prior MAE")
            for arm, bundle in fitted[target].items():
                pred = ensemble(bundle["models"], x) * scale + median
                placebo = placebo_ensemble(bundle["placebos"], x, pack["subjects"], f"robust_external:{db}:{target}") * scale + median
                matched = placebo_ensemble(bundle["matched_placebos"], x, pack["subjects"], f"robust_external:{db}:{target}") * scale + median
                result = arm_score(selected, y, pred, placebo, median, f"robust:{db}:{target}:{arm}")
                result["vs_matched_complexity_placebo"] = paired_screen(selected, y, pred, matched, f"matched:{db}:{target}:{arm}")
                result["matched_complexity_placebo_mae"] = float(np.abs(matched - y).mean())
                result["diagnostic_pass"] &= result["vs_matched_complexity_placebo"]["pass"]
                item["arms"][arm] = result
                predictions[arm] = pred
            chosen = report["source_selection"][target]["chosen_arm"]
            predictions["source_selected"] = predictions[chosen]
            item["arms"]["source_selected"] = {**item["arms"][chosen], "chosen_arm": chosen}
            item["paired_diagnostics"] = {arm: {
                base: paired_screen(selected, y, pred, predictions[base], f"robust_pair:{db}:{target}:{arm}:{base}")
                for base in ("latent_erm32", "latent_erm")}
                for arm, pred in predictions.items() if arm in fitted[target] or arm == "source_selected"}
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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/masked_robust_readouts_20260903"))
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
        raise ValueError("incompatible prior experiment")
    inputs = {str(p): sha256(p) for p in (args.eicu, args.mimic_current, args.nwicu_current)}
    if any(prior["input_hashes"].get(p) != h for p, h in inputs.items()):
        raise ValueError("input cohorts changed from frozen reference")
    previous = joblib.load(args.previous / "frozen_probes.joblib")
    if previous["release_binding"] != binding or previous["selection_scope"] != "eicu_tuning_patients_only":
        raise ValueError("incompatible reference head bundle")
    report = {"schema": "whole_body_jepa.masked_robust_readouts.v1", "release_binding": binding,
              "input_hashes": inputs, "previous_report_sha256": sha256(args.previous / "report.json"),
              "previous_weights_sha256": sha256(args.previous / "frozen_probes.joblib"),
              "code_hashes": {str(p): sha256(p) for p in (Path(__file__), Path("scripts/masked_robust_readouts.py"))},
              "protocol_sha256": sha256(Path("MASKED_ROBUST_READOUT_PROTOCOL_20260903.md")),
              "protocol": {"families": FAMILIES, "seeds": SEEDS, "equal_capacity_bagged_controls": True,
                           "selection": "eICU tune: half mean plus half worst supported care-unit MAE",
                           "tail_bounds": "fit-only truth 10th and 90th percentiles; reporting only",
                           "frozen_shared_encoder": True, "source_roles_unchanged": True},
              "support": {}, "training": {}, "source_selection": {}, "evaluation": {},
              "complete": False, "promotion_allowed": False, "runtime_modified": False,
              "forecast_unchanged_by_construction": True, "untouched_lockbox_claim": False,
              "next_measurement_time_modeled": False, "causal_claim_allowed": False}
    args.output_dir.mkdir(parents=True)
    targets = sorted({c["target"] for c in assets.source_selector["cells"].values()} - {"hypoxemia_event"})
    frame, arrays, roles, packs, hidden = source_setup(args, assets, targets, report)
    if report["source_roles"] != prior["source_roles"]:
        raise ValueError("source split changed")
    del frame, arrays, roles
    fitted = {}
    dump(report, args.output_dir)
    for target, pack in packs.items():
        fitted[target], info = {}, {}
        idx = assets.checkpoint["variables"].index(target)
        report["support"][target]["physical_tail_bounds"] = (
            np.quantile(pack["fit"]["y"], [.1, .9]) * assets.scaler.scales[idx] + assets.scaler.medians[idx]).tolist()
        for family in FAMILIES:
            bundle, info[family] = fit_family(family, pack["fit"], pack["tune"], target)
            if bundle is not None:
                fitted[target][family] = bundle
            print(f"fit {target} {family}: {info[family]['status']} selected={info[family].get('selected')}", flush=True)
        report["training"][target] = info
        report["source_selection"][target] = choose_source_head(info, prior["training"][target])
        dump(report, args.output_dir)
    joblib.dump({"release_binding": binding, "models": fitted, "source_selection": report["source_selection"],
                 "status": "candidate_only"}, args.output_dir / "readouts.joblib")
    report["readouts_sha256"] = sha256(args.output_dir / "readouts.joblib")
    report["selection_frozen_before_external_evaluation"] = True
    dump(report, args.output_dir)
    del packs
    evaluate(args, assets, hidden, fitted, previous["models"], prior, report)
    if binding != {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}:
        raise ValueError("serving artifacts changed")
    report["complete"] = True
    dump(report, args.output_dir)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
