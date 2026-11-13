"""Four controlled generalization diagnostics; never modify serving assets."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.audit_masked_readout_ensemble import cohort_coverage
from scripts.audit_release_masked_cells import donor_indices, infer, read_examples, score, select_rows, valid_future_rows
from scripts.masked_generalization_methods import (
    ALPHAS, STEPS, cached_context, ensemble_predict, fit_neural_method, fit_ridge_probe,
    group_weights, raw_view, selection_score, source_roles,
)
from scripts.train_release_masked_abc import batch_inputs, candidate_inputs, paired_screen
from whole_body_jepa.multisource_intervals import SEEDS, sha256, stable_hash
from whole_body_jepa.nowcast import related_siblings
from whole_body_jepa.runtime import load_release
from whole_body_jepa.source_selection import PERSISTENCE


JOINT_STEPS = (8, 32, 64)


def dump(report, directory):
    report["summary"] = {
        db: {arm: [t for t, task in tasks.items()
                   if task.get("arms", {}).get(arm, {}).get("diagnostic_pass")]
             for arm in sorted({a for task in tasks.values() for a in task.get("arms", {})})}
        for db, tasks in report["evaluation"].items()}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))


def make_pack(frame, arrays, rows, target, assets, hidden):
    idx = assets.checkpoint["variables"].index(target)
    latent, pre = cached_context(assets.model, arrays, rows, hidden)
    selected = frame.iloc[rows]
    return {"latent": latent, "pre": pre, "raw": raw_view(arrays, rows, hidden),
            "y": torch.from_numpy(arrays["current"][rows, idx]).float(),
            "subjects": selected.subject_id.to_numpy(), "units": selected.careunit.to_numpy(),
            "rows": rows, "frame": selected}


def source_setup(args, assets, targets, report):
    frame = read_examples(args.eicu, assets, "eicu", 3, grid=True)
    arrays = candidate_inputs(frame, assets, 3)
    roles = source_roles(frame)
    report["source_roles"] = {name: cohort_coverage(frame.loc[mask]) for name, mask in roles.items()}
    variables = assets.checkpoint["variables"]
    packs, hidden = {}, {}
    for target in targets:
        hidden[target] = sorted(variables.index(v) for v in ({target} | related_siblings(target)) & set(variables))
        valid = frame[f"{target}_t"].notna() & frame[f"{target}_age_hr"].between(0, 1)
        rows = {r: select_rows(frame, valid & roles[r], 3000, target) for r in ("fit", "tune")}
        report["support"][target] = {r: cohort_coverage(frame.iloc[ix]) for r, ix in rows.items()}
        report["support"][target]["hidden_variables"] = [variables[i] for i in hidden[target]]
        if min(map(len, rows.values())) < 100:
            report["support"][target]["status"] = "insufficient_source_support"
            continue
        packs[target] = {r: make_pack(frame, arrays, ix, target, assets, hidden[target]) for r, ix in rows.items()}
    return frame, arrays, roles, packs, hidden


def train_frozen(assets, packs, report, directory):
    bundles = {}
    for target, pack in packs.items():
        idx = assets.checkpoint["variables"].index(target)
        initial = assets.model.reconstruction_heads[idx]
        item = bundles[target] = {}
        info = report["training"].setdefault(target, {})
        for view in ("raw", "latent"):
            arm = view + "_ridge"
            item[arm] = fit_ridge_probe(pack["fit"], pack["tune"], view, target)
            info[arm] = {k: v for k, v in item[arm].items() if k not in {"model", "placebos"}}
        for method in ("erm", "balanced", "dro"):
            arm = "latent_" + method
            item[arm] = fit_neural_method(initial, pack["fit"], pack["tune"], method, target)
            info[arm] = {k: v for k, v in item[arm].items() if not k.endswith("heads")}
            print(f"fit {target} {method}: chosen {info[arm]['chosen_steps']} updates", flush=True)
        dump(report, directory)
    joblib.dump({"release_binding": report["release_binding"], "models": bundles,
                 "status": "candidate_only", "selection_scope": "eicu_tuning_patients_only"},
                directory / "frozen_probes.joblib")
    return bundles


def shared_tune(model, packs):
    details = {}
    with torch.no_grad():
        for target, pack in packs.items():
            pred = model.reconstruction_heads[pack["index"]](model.state_encoder(pack["tune"]["pre"])).flatten().numpy()
            details[target] = selection_score(pack["tune"]["y"], pred, pack["tune"]["units"])
    return {"target_scores": details, "selection_score": float(np.mean([v["selection_score"] for v in details.values()]))}


def replay_loss(model, teacher, arrays, rows, horizon):
    inputs = batch_inputs(arrays, rows)
    h = torch.full((len(rows),), float(horizon))
    with torch.no_grad():
        expected = teacher(*inputs, h)
    actual = model(*inputs, h)
    # Preserve both direct values and the latent features consumed by external
    # shared readouts. No future measurements or retrospective actions are used.
    return sum(F.smooth_l1_loss(actual[key], expected[key]) for key in
               ("value", "context_latent", "predicted_target_latent")) / 3


def train_shared(assets, frame, arrays, roles, packs, report, directory):
    teacher = assets.model.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for target, pack in packs.items():
        pack["index"] = assets.checkpoint["variables"].index(target)
    retention = select_rows(frame, roles["fit"], 1500, "replay")
    targets = sorted(packs)
    candidates = {}
    for protected in (False, True):
        arm = "shared_protected" if protected else "shared_unprotected"
        report["shared_training"][arm] = {}
        for placebo in (False, True):
            kind = "placebo" if placebo else "real"
            model = copy.deepcopy(teacher).eval()
            for name, p in model.named_parameters():
                p.requires_grad_(name.startswith(("state_encoder.", "reconstruction_heads.")))
            optimizer = torch.optim.AdamW([
                {"params": model.state_encoder.parameters(), "lr": .0001},
                {"params": model.reconstruction_heads.parameters(), "lr": .003},
            ], weight_decay=.001)
            rng = np.random.default_rng(73)
            working = {}
            for target, pack in packs.items():
                fit, tune = dict(pack["fit"]), dict(pack["tune"])
                if placebo:
                    for role, part in (("fit", fit), ("tune", tune)):
                        ix = donor_indices(part["subjects"], stable_hash("shared_placebo", role, target))
                        part["pre"] = part["pre"][ix]
                working[target] = {"fit": fit, "tune": tune, "index": pack["index"]}
            snapshots, curve = {}, {}
            for cycle in range(1, max(JOINT_STEPS) + 1):
                schedule = rng.permutation(len(targets))
                rec_losses, replay_losses = [], []
                for ti in schedule:
                    target = targets[ti]
                    data = working[target]["fit"]
                    weights, _, _ = group_weights(data["units"])
                    rows = rng.choice(len(data["y"]), 128, replace=True, p=weights)
                    rr = rng.choice(retention, 16, replace=len(retention) < 16)
                    h = (1, 3, 6, 12, 24, 48)[int(rng.integers(6))]
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model.reconstruction_heads[working[target]["index"]](model.state_encoder(data["pre"][rows])).flatten()
                    reconstruction = F.l1_loss(prediction, data["y"][rows])
                    preserve = replay_loss(model, teacher, arrays, rr, h) if protected else reconstruction.new_zeros(())
                    loss = reconstruction + 10. * preserve
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("nonfinite shared-state objective")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                    optimizer.step()
                    rec_losses.append(float(reconstruction.detach()))
                    replay_losses.append(float(preserve.detach()))
                if cycle in JOINT_STEPS:
                    item = shared_tune(model, working)
                    item.update(train_normalized_l1=float(np.mean(rec_losses)), replay_loss=float(np.mean(replay_losses)))
                    curve[cycle] = item
                    snapshots[cycle] = {k: v.detach().clone() for k, v in model.state_dict().items()
                                        if k.startswith(("state_encoder.", "reconstruction_heads."))}
                    print(f"shared {arm} {kind} updates/target={cycle} tune={item['selection_score']:.5g}", flush=True)
            chosen = min(JOINT_STEPS, key=lambda s: (curve[s]["selection_score"], s))
            changed = snapshots[chosen]
            final_state = teacher.state_dict().copy()
            final_state.update(changed)
            model.load_state_dict(final_state, strict=True)
            if any(not torch.equal(v, teacher.state_dict()[k]) for k, v in model.state_dict().items()
                   if not k.startswith(("state_encoder.", "reconstruction_heads."))):
                raise AssertionError("limited state update changed frozen layers")
            path = directory / f"{arm}_{kind}.pt"
            torch.save({"state_dict": model.state_dict(), "base_encoder_sha256": report["release_binding"]["whole_body_joint_jepa.pt"],
                        "status": "candidate_only", "chosen_steps": chosen}, path)
            report["shared_training"][arm][kind] = {"chosen_steps": chosen, "curve": curve, "sha256": sha256(path),
                                                       "optimizer_seed": 73, "scope": "body_projection_and_reconstruction_only"}
            candidates[arm + ("_placebo" if placebo else "")] = replace(assets, model=model)
            dump(report, directory)
    return candidates


def arm_score(frame, y, pred, placebo, prior, salt):
    scored = score(frame, y, pred, prior, salt)
    matched = paired_screen(frame, y, pred, placebo, salt + ":trained_placebo")
    return {"mae": float(np.abs(pred - y).mean()), "trained_placebo_mae": float(np.abs(placebo - y).mean()),
            "screen": scored, "vs_trained_placebo": matched,
            "diagnostic_pass": scored["diagnostic_pass"] and matched["pass"]}


def evaluate(args, assets, targets, hidden, bundles, shared, report):
    for db, path, grid in (("eicu_hospital", args.eicu, True), ("mimic", args.mimic_current, False),
                           ("nwicu", args.nwicu_current, False)):
        frame = read_examples(path, assets, "eicu" if grid else db, 3, grid=grid)
        arrays = candidate_inputs(frame, assets, 3)
        eligible_role = source_roles(frame)["hospital_test"] if grid else np.ones(len(frame), bool)
        report["evaluation"][db] = {}
        for target in targets:
            rows = select_rows(frame, eligible_role & frame[f"{target}_t"].notna() & frame[f"{target}_age_hr"].between(0, 1),
                               args.max_patients, target)
            item = report["evaluation"][db][target] = {"patients": len(rows), "arms": {}}
            if len(rows) < 200 or target not in bundles:
                item["status"] = "insufficient_support"
                continue
            item["status"] = "evaluated"
            pack = make_pack(frame, arrays, rows, target, assets, hidden[target])
            idx = assets.checkpoint["variables"].index(target)
            median, scale = float(assets.scaler.medians[idx]), float(assets.scaler.scales[idx])
            y = frame.iloc[rows][f"{target}_t"].to_numpy()
            chosen = frame.iloc[rows]
            all_predictions = {}
            for arm, bundle in bundles[target].items():
                if arm.endswith("ridge"):
                    view = arm.removesuffix("_ridge")
                    pred = bundle["model"].predict(pack[view])
                    controls = []
                    for seed, model in zip(SEEDS, bundle["placebos"]):
                        p = donor_indices(pack["subjects"], stable_hash("external_placebo", db, target, seed))
                        controls.append(model.predict(np.asarray(pack[view])[p]))
                    placebo = np.mean(controls, axis=0)
                else:
                    pred = ensemble_predict(bundle["heads"], pack["latent"])
                    controls = []
                    for seed, model in zip(SEEDS, bundle["placebo_heads"]):
                        p = donor_indices(pack["subjects"], stable_hash("external_placebo", db, target, seed))
                        controls.append(ensemble_predict([model], pack["latent"][p]))
                    placebo = np.mean(controls, axis=0)
                    if arm == "latent_erm":
                        first = ensemble_predict(bundle["fixed32_heads"], pack["latent"]) * scale + median
                        first_controls = []
                        for seed, model in zip(SEEDS, bundle["fixed32_placebo_heads"]):
                            p = donor_indices(pack["subjects"], stable_hash("external_placebo", db, target, seed))
                            first_controls.append(ensemble_predict([model], pack["latent"][p]))
                        item["arms"]["latent_erm32"] = arm_score(chosen, y, first, np.mean(first_controls, axis=0) * scale + median,
                                                                  median, f"{db}:{target}:erm32")
                        all_predictions["latent_erm32"] = first
                pred, placebo = pred * scale + median, placebo * scale + median
                all_predictions[arm] = pred
                item["arms"][arm] = arm_score(chosen, y, pred, placebo, median, f"{db}:{target}:{arm}")
            for arm in ("shared_unprotected", "shared_protected"):
                with torch.no_grad():
                    model = shared[arm].model
                    pred = model.reconstruction_heads[idx](model.state_encoder(pack["pre"])).flatten().numpy() * scale + median
                    placebo_model = shared[arm + "_placebo"].model
                    p = donor_indices(pack["subjects"], stable_hash("shared_placebo", db, target))
                    placebo = placebo_model.reconstruction_heads[idx](placebo_model.state_encoder(pack["pre"][p])).flatten().numpy() * scale + median
                all_predictions[arm] = pred
                item["arms"][arm] = arm_score(chosen, y, pred, placebo, median, f"{db}:{target}:{arm}")
            item["paired_diagnostics"] = {
                f"{a}_vs_{b}": paired_screen(chosen, y, all_predictions[a], all_predictions[b], f"{db}:{target}:{a}:{b}")
                for a, b in (("raw_ridge", "latent_ridge"), ("latent_erm", "latent_erm32"),
                             ("latent_balanced", "latent_erm"), ("latent_dro", "latent_erm"),
                             ("shared_protected", "shared_unprotected"))}
            print(f"evaluated {db} {target}: " + " ".join(f"{a}={r['mae']:.4g}({r['diagnostic_pass']})" for a, r in item["arms"].items()), flush=True)
            dump(report, args.output_dir)


def forecast_retention(args, assets, shared, report):
    cells = assets.source_selector["cells"]
    for horizon in sorted({int(v["horizon_hours"]) for v in cells.values()}):
        for db in ("mimic", "nwicu"):
            path = Path(args.mimic_template.format(horizon=horizon)) if db == "mimic" else args.nwicu_future
            frame = read_examples(path, assets, db, horizon, grid=db == "nwicu")
            arrays = candidate_inputs(frame, assets, horizon)
            for cell, config in cells.items():
                if int(config["horizon_hours"]) != horizon:
                    continue
                target = config["target"]
                item = report["forecast_retention"].setdefault(db, {})[cell] = {"source": config["source_model"]}
                if target == "hypoxemia_event" or f"future_{target}" not in frame:
                    item["status"] = "separate_event_or_missing_label"
                    continue
                valid = valid_future_rows(frame, target)
                if db == "mimic" and cell in assets.multisource_heads["heads"]:
                    name = cell.replace("@", "_")
                    cache, metadata = args.previous_cache / f"{name}_mimic.parquet", args.previous_cache / f"{name}.json"
                    if not cache.exists() or not metadata.exists():
                        item["status"] = "training_exclusions_unavailable"
                        continue
                    meta = json.loads(metadata.read_text())["key"]
                    if meta["binding"] != assets.source_selector["release_binding"] or meta["mimic_sha256"] != sha256(path):
                        raise ValueError("incompatible exclusion cache")
                    subjects = pd.read_parquet(cache, columns=["subject_id"]).subject_id.astype(str).str.replace("MIMIC:", "mimic:", regex=False)
                    valid &= frame.subject_id.isin(set(subjects))
                rows = select_rows(frame, valid, args.max_patients, cell)
                if len(rows) < 200:
                    item.update(status="insufficient_support", patients=len(rows))
                    continue
                selected = frame.iloc[rows]
                y = selected[f"future_{target}"].to_numpy()
                persistence = selected[f"{target}_t"].to_numpy()
                source = config["source_model"]
                base = persistence if source == PERSISTENCE else infer(assets, arrays, rows, target, cell)[source]
                item.update(status="evaluated", patients=len(rows), baseline_mae=float(np.abs(base - y).mean()),
                            frozen_readout_forecasts_unchanged_by_construction=True, arms={})
                for arm in ("shared_unprotected", "shared_protected"):
                    pred = persistence if source == PERSISTENCE else infer(shared[arm], arrays, rows, target, cell)[source]
                    item["arms"][arm] = {"mae": float(np.abs(pred - y).mean()),
                                          "max_absolute_change": float(np.max(np.abs(pred - base))),
                                          "vs_incumbent": paired_screen(selected, y, pred, base, f"retention:{db}:{cell}:{arm}")}
                print(f"retention {db} {cell}: A={item['baseline_mae']:.4g} " + " ".join(f"{a}={v['mae']:.4g}" for a, v in item["arms"].items()), flush=True)
                dump(report, args.output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--eicu", type=Path, default=Path("/private/tmp/eicu_hospital_balanced_examples_3h_20260903.parquet"))
    parser.add_argument("--mimic-current", type=Path, default=Path("/private/tmp/mimiciv_observation_transitions_3h_full_v31_treatment_unit_safe.parquet"))
    parser.add_argument("--nwicu-current", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet"))
    parser.add_argument("--mimic-template", default="/private/tmp/mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe.parquet")
    parser.add_argument("--nwicu-future", type=Path, default=Path("/private/tmp/nwicu_event_examples_41_grid.parquet"))
    parser.add_argument("--previous-cache", type=Path, default=Path("/private/tmp/unified_source_predictions_20260902"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/masked_generalization_20260903"))
    parser.add_argument("--max-patients", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(73)
    torch.use_deterministic_algorithms(True)
    if args.output_dir.exists():
        raise ValueError("use a fresh experiment directory")
    assets = load_release(args.model_dir)
    binding = {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}
    targets = sorted({v["target"] for v in assets.source_selector["cells"].values()} - {"hypoxemia_event"})
    paths = {args.eicu, args.mimic_current, args.nwicu_current, args.nwicu_future}
    paths.update(Path(args.mimic_template.format(horizon=int(c["horizon_hours"]))) for c in assets.source_selector["cells"].values())
    report = {"schema": "whole_body_jepa.masked_generalization_diagnostics.v1", "release_binding": binding,
              "input_hashes": {str(p): sha256(p) for p in sorted(paths)},
              "code_hashes": {str(p): sha256(p) for p in (Path(__file__), Path("scripts/masked_generalization_methods.py"))},
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "protocol": {"head_steps": list(STEPS), "ridge_alphas": list(ALPHAS), "seeds": list(SEEDS),
                           "shared_optimizer_seed": 73, "shared_steps_per_target": list(JOINT_STEPS),
                           "head_weight_decay": .001, "head_lr": .003, "minimum_training_unit_support": 20,
                           "selection": "eICU tuning only: half mean + half worst supported care-unit normalized MAE",
                           "heldout_hospitals": "sha256(masked_generalization_hospital,hospitalid)%5==0",
                           "raw_probe_is_not_capacity_matched_to_latent": True,
                           "existing_encoder_may_have_seen_source_test_patients": True,
                           "no_reused_forecast_intervals": True},
              "support": {}, "training": {}, "shared_training": {}, "evaluation": {}, "forecast_retention": {},
              "complete": False, "runtime_modified": False, "promotion_allowed": False,
              "causal_claim_allowed": False, "untouched_lockbox_claim": False,
              "next_measurement_time_modeled": False}
    args.output_dir.mkdir(parents=True)
    dump(report, args.output_dir)
    frame, arrays, roles, packs, hidden = source_setup(args, assets, targets, report)
    dump(report, args.output_dir)
    bundles = train_frozen(assets, packs, report, args.output_dir)
    shared = train_shared(assets, frame, arrays, roles, packs, report, args.output_dir)
    report["selection_frozen_before_external_evaluation"] = True
    dump(report, args.output_dir)
    del frame, arrays, packs
    evaluate(args, assets, targets, hidden, bundles, shared, report)
    forecast_retention(args, assets, shared, report)
    if binding != {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}:
        raise ValueError("serving assets changed")
    report["complete"] = True
    dump(report, args.output_dir)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
