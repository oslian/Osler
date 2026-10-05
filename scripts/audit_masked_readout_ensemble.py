"""Frozen shared-state readouts: seven fit seeds and a trained matched placebo.

This extends the preregistered B intervention, not the serving forecast graph.
All readouts consume the SAME frozen context latent. No new body encoder.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_release_masked_cells import donor_indices, read_examples, score, select_rows
from scripts.train_release_masked_abc import batch_inputs, candidate_inputs, paired_screen
from whole_body_jepa.multisource_intervals import SEEDS, patient_split, sha256, stable_hash
from whole_body_jepa.nowcast import related_siblings
from whole_body_jepa.runtime import load_release


def encode_masked(model, arrays, rows, hidden):
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            inputs = batch_inputs(arrays, rows[start:start + 256], hidden)
            pieces.append(model.encode_context_components(*inputs)[0].numpy())
    return torch.from_numpy(np.concatenate(pieces))


def fit_readout(initial, state, label, seed, steps=32, batch_size=128, lr=.003):
    # Cache the frozen state once; optimization never has access to an encoder.
    head = copy.deepcopy(initial).eval()
    for p in head.parameters():
        p.requires_grad_(True)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        rows = rng.choice(len(state), batch_size, replace=len(state) < batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = F.l1_loss(head(state[rows]).flatten(), label[rows])
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite readout loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
    return head


def predict_readouts(heads, state, median, scale):
    with torch.inference_mode():
        return np.stack([head(state).flatten().numpy() * scale + median for head in heads])


def write_report(report, directory):
    summary = {}
    for db, tasks in report["evaluation"].items():
        summary[db] = {
            "evaluated": sum(t.get("status") == "evaluated" for t in tasks.values()),
            "diagnostic_pass": [t for t, v in tasks.items() if v.get("diagnostic_pass")],
            "nominal_seven_patient_split_pass": [t for t, v in tasks.items()
                if v.get("status") == "evaluated" and all(
                    all(s["pass"] for s in c["patient_splits"])
                    for c in v["ensemble"]["comparisons"].values())
                and all(s["pass"] for s in v["vs_trained_placebo"]["patient_splits"])],
        }
    report["summary"] = summary
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))


def cohort_coverage(frame):
    """Aggregate audit metadata only; never concatenate it into model inputs."""
    result = {"patients": int(frame.subject_id.nunique()),
              "encounters": int(frame.stay_id.nunique()), "rows": len(frame)}
    if "hospitalid" in frame:
        counts = frame.groupby("hospitalid").stay_id.nunique()
        result.update(hospitals=len(counts), largest_hospital_encounter_fraction=(
            float(counts.max() / counts.sum()) if len(counts) else None))
    result["care_unit_patients"] = {
        str(k): int(v) for k, v in frame.groupby("careunit").subject_id.nunique().items()}
    return result


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    binding = {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}
    if args.output_dir.exists():
        raise ValueError("experiment output already exists")
    args.output_dir.mkdir(parents=True)
    targets = sorted({v["target"] for v in assets.source_selector["cells"].values()} - {"hypoxemia_event"})
    variables = assets.checkpoint["variables"]
    frame = read_examples(args.eicu, assets, "eicu", 3, grid=True)
    arrays = candidate_inputs(frame, assets, 3)
    discovery, _ = patient_split(frame["subject_id"], 73)
    hidden = {t: sorted(variables.index(v) for v in ({t} | related_siblings(t)) & set(variables)) for t in targets}
    report = {"schema": "whole_body_jepa.frozen_readout_seed_ensemble.v1", "release_binding": binding,
              "script_sha256": sha256(Path(__file__)), "seed_list": list(SEEDS), "source_split_seed": 73,
              "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "training_input_sha256": sha256(args.eicu), "training": {}, "evaluation": {},
              "complete": False, "runtime_modified": False, "promotion_allowed": False,
              "causal_claim_allowed": False, "next_measurement_time_modeled": False,
              "untouched_lockbox_claim": False,
              "protocol": "Fixed eICU source-only heads; equal mean of seven optimizer seeds. Exact-capacity placebo trained on deranged patient contexts, independently deranged at test. Fresh truth <=1h, target+sibling full-history masking. Seven patient evaluation splits, all supported careunits. No tuning on MIMIC/NWICU. No reuse of forecast intervals."}
    report["source_cohort"] = cohort_coverage(frame)
    report["discovery_cohort"] = cohort_coverage(frame.loc[discovery])
    fitted, placebos = {}, {}
    for target in targets:
        idx = variables.index(target)
        valid = frame[f"{target}_t"].notna() & frame[f"{target}_age_hr"].between(0, 1) & discovery
        rows = select_rows(frame, valid, 3000, target)
        if len(rows) < 100:
            report["training"][target] = {"status": "insufficient_source_support"}
            continue
        state = encode_masked(assets.model, arrays, rows, hidden[target])
        y = torch.from_numpy(arrays["current"][rows, idx]).float()
        subjects = frame.iloc[rows]["subject_id"].to_numpy()
        fitted[target], placebos[target] = [], []
        for seed in SEEDS:
            draw_seed = stable_hash("readout_ensemble", target, seed)
            fitted[target].append(fit_readout(assets.model.reconstruction_heads[idx], state, y, draw_seed))
            permutation = donor_indices(subjects, stable_hash("train_placebo", target, seed))
            placebos[target].append(fit_readout(assets.model.reconstruction_heads[idx], state[permutation], y, draw_seed))
        report["training"][target] = {"patients": len(rows), "steps_per_seed": 32,
                                      "parameters_per_head": sum(p.numel() for p in fitted[target][0].parameters()),
                                      "hidden_variables": [variables[i] for i in hidden[target]]}
        print(f"fit frozen readouts {target}: {len(rows)} patients, seven paired model/placebo fits", flush=True)
    del frame, arrays
    checkpoint = {"base_encoder_sha256": binding["whole_body_joint_jepa.pt"],
                  "mask_mode": "neutralized", "reconstruction_latent": "anchor_context",
                  "aggregation": "equal_mean", "seeds": list(SEEDS), "promotion_status": "candidate_only",
                  "heads": {t: [h.state_dict() for h in hs] for t, hs in fitted.items()},
                  "placebo_heads": {t: [h.state_dict() for h in hs] for t, hs in placebos.items()}}
    torch.save(checkpoint, args.output_dir / "readouts.pt")
    report["checkpoint_sha256"] = sha256(args.output_dir / "readouts.pt")
    write_report(report, args.output_dir)
    for db, path, grid in (("eicu", args.eicu, True), ("mimic", args.mimic_current, False),
                            ("nwicu", args.nwicu_current, False)):
        report.setdefault("input_hashes", {})[str(path)] = sha256(path)
        frame = read_examples(path, assets, db, 3, grid=grid)
        arrays = candidate_inputs(frame, assets, 3)
        tasks = report["evaluation"].setdefault(db, {})
        for target in targets:
            idx = variables.index(target)
            valid = frame[f"{target}_t"].notna() & frame[f"{target}_age_hr"].between(0, 1)
            if db == "eicu":
                _, development = patient_split(frame["subject_id"], 73)
                valid &= development
            rows = select_rows(frame, valid, args.max_patients, target)
            item = {"patients": len(rows)}
            tasks[target] = item
            if len(rows) < 200 or target not in fitted:
                item["status"] = "insufficient_support"
                continue
            state = encode_masked(assets.model, arrays, rows, hidden[target])
            y = frame.iloc[rows][f"{target}_t"].to_numpy()
            median, scale = float(assets.scaler.medians[idx]), float(assets.scaler.scales[idx])
            predictions = predict_readouts(fitted[target], state, median, scale)
            placebo_predictions = []
            for i, seed in enumerate(SEEDS):
                donors = donor_indices(frame.iloc[rows]["subject_id"], stable_hash("test_placebo", target, db, seed))
                placebo_predictions.append(predict_readouts([placebos[target][i]], state[donors], median, scale)[0])
            ensemble = predictions.mean(axis=0)
            placebo = np.mean(placebo_predictions, axis=0)
            baseline = predict_readouts([assets.model.reconstruction_heads[idx]], state, median, scale)[0]
            item.update(status="evaluated", baseline_mae=float(np.abs(baseline - y).mean()),
                        trained_placebo_mae=float(np.abs(placebo - y).mean()),
                        ensemble=score(frame.iloc[rows], y, ensemble, median, f"{db}:{target}:current"),
                        vs_baseline=paired_screen(frame.iloc[rows], y, ensemble, baseline, f"ensemble:{db}:{target}:A"),
                        vs_trained_placebo=paired_screen(frame.iloc[rows], y, ensemble, placebo, f"ensemble:{db}:{target}:placebo"),
                        optimizer_seed_results={str(seed): {"mae": float(np.abs(predictions[i] - y).mean()),
                            "vs_population": paired_screen(frame.iloc[rows], y, predictions[i], np.full(len(y), median),
                                                           f"single:{db}:{target}:{seed}")}
                            for i, seed in enumerate(SEEDS)})
            item["diagnostic_pass"] = item["ensemble"]["diagnostic_pass"] and item["vs_trained_placebo"]["pass"]
            print(f"ensemble {db} {target}: A={item['baseline_mae']:.4g} E={item['ensemble']['mae']:.4g} "
                  f"P={item['trained_placebo_mae']:.4g} pass={item['diagnostic_pass']}", flush=True)
            write_report(report, args.output_dir)
    if binding != {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}:
        raise ValueError("release changed")
    report["complete"] = True
    write_report(report, args.output_dir)
    print(json.dumps(report["summary"]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/masked_readout_ensemble_20260903"))
    p.add_argument("--eicu", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet"))
    p.add_argument("--mimic-current", type=Path, default=Path("/private/tmp/mimiciv_observation_transitions_3h_full_v31_treatment_unit_safe.parquet"))
    p.add_argument("--nwicu-current", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet"))
    p.add_argument("--max-patients", type=int, default=4000)
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
