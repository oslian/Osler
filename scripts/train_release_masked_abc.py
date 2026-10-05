"""Controlled, source-only repair of the packaged JEPA reconstruction branch.

A: frozen release. B: anchor readout only. C0: joint neutral-mask training.
C: identical joint training with explicit missing-value masks. No serving edits.
The external screens are diagnostics, never an automatic promotion procedure.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.audit_release_masked_cells import (
    infer, input_frame, read_examples, score, select_rows, valid_future_rows,
)
from whole_body_jepa.multisource_intervals import SEEDS, _bootstrap, patient_split, sha256, stable_hash
from whole_body_jepa.nowcast import related_siblings
from whole_body_jepa.runtime import _build_inputs, load_release
from whole_body_jepa.source_selection import BASE, MULTI, PERSISTENCE


ARMS = ("A", "B", "C0", "C")
CONTEXT_PREFIXES = (
    "belief_filter.", "variable_embedding", "module_token_embedding.",
    "token_projection.", "variable_mixer.", "module_mixer.",
    "presence_projection.", "body_pool_projection.", "state_encoder.",
)


def candidate_inputs(frame, assets, horizon, *, explicit=False):
    view = input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours)
    arrays = _build_inputs(view, assets, horizon)
    if explicit:
        # Only the marker changes; ages, unit identity and organ presence remain
        # neutral in both arms. An observed physical zero is still observed.
        _, mask, _ = assets.scaler.normalize_current(view)
        arrays["mask"] = mask
        n = len(assets.checkpoint["variables"])
        arrays["input"][:, n:2 * n] = mask
    return arrays


def batch_inputs(arrays, rows, hidden=(), *, explicit=False):
    n = arrays["current"].shape[1]
    obs = torch.from_numpy(arrays["input"][arrays["history"][rows]].copy()).float()
    current = torch.from_numpy(arrays["current"][rows].copy()).float()
    mask = torch.from_numpy(arrays["mask"][rows].copy()).float()
    if len(hidden):
        obs[..., list(hidden)] = 0
        current[:, list(hidden)] = 0
        if explicit:
            obs[..., [n + i for i in hidden]] = 0
            mask[:, list(hidden)] = 0
    return (obs, current, mask,
            torch.from_numpy(arrays["ages"][rows]).float(),
            torch.from_numpy(arrays["presence"][rows]).float())


def masked_arrays(arrays, hidden, explicit):
    result = dict(arrays)
    for key in ("input", "current", "mask"):
        result[key] = arrays[key].copy()
    result["input"][:, hidden] = 0
    result["current"][:, hidden] = 0
    if explicit:
        result["input"][:, [arrays["current"].shape[1] + i for i in hidden]] = 0
        result["mask"][:, hidden] = 0
    return result


def configure_training(model, arm):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("reconstruction_heads.")
            or (arm in {"C0", "C"} and name.startswith(CONTEXT_PREFIXES))
        )
    # Equal deterministic readout experiments, not a dropout augmentation test.
    model.eval()
    return [p for p in model.parameters() if p.requires_grad]


def components_loss(student, teacher):
    # Preserve the inputs to ALL shipped readouts, including belief and organ
    # features, not merely one headline forecast. No future labels enter here.
    return sum(F.smooth_l1_loss(x, y.detach()) for x, y in zip(student[:4], teacher[:4])) / 4


def paired_training_rows(rng, pool, retention_pool, batch_size):
    # Consume identical randomness even when B does not use retention loss.
    return (rng.choice(pool, batch_size, replace=len(pool) < batch_size),
            rng.choice(retention_pool, batch_size, replace=len(retention_pool) < batch_size))


def paired_screen(frame, truth, candidate, reference, salt):
    if len(frame) < 200:
        return {"pass": False, "status": "insufficient_support"}
    delta = np.abs(candidate - truth) - np.abs(reference - truth)
    splits = []
    for seed in SEEDS:
        _, test = patient_split(frame["subject_id"], seed)
        result = _bootstrap(delta[test], stable_hash(salt, seed)) if test.sum() >= 100 else {"pass": False}
        splits.append({"seed": seed, **result})
    units = frame["careunit"].astype(str).to_numpy()
    unit_scores = {u: _bootstrap(delta[units == u], stable_hash(salt, u))
                   for u in sorted(set(units))
                   if u.lower() not in {"unknown", "nan", "none", ""} and np.sum(units == u) >= 30}
    return {"all": _bootstrap(delta, stable_hash(salt, "all")), "patient_splits": splits,
            "careunits": unit_scores,
            "pass": all(v["pass"] for v in splits) and len(unit_scores) >= 2
                    and all(v["pass"] for v in unit_scores.values())}


def train(args, assets, targets, report):
    frame = read_examples(args.eicu, assets, "eicu", 3, grid=True)
    variables = assets.checkpoint["variables"]
    neutral = candidate_inputs(frame, assets, 3)
    explicit = candidate_inputs(frame, assets, 3, explicit=True)
    discovery, development = patient_split(frame["subject_id"], args.seed)
    assert not set(frame.loc[discovery, "subject_id"]) & set(frame.loc[development, "subject_id"])
    pools, hidden = {}, {}
    for target in targets:
        eligible = frame[f"{target}_t"].notna() & frame[f"{target}_age_hr"].between(0, 1)
        pools[target] = select_rows(frame, eligible & discovery, args.max_train_patients, target)
        hidden[target] = sorted(variables.index(v) for v in ({target} | related_siblings(target)) & set(variables))
    usable = [t for t in targets if len(pools[t]) >= 100]
    report["training_support"] = {t: len(pools[t]) for t in targets}
    report["source_patients"] = {"discovery": int(frame.loc[discovery, "subject_id"].nunique()),
                                 "development": int(frame.loc[development, "subject_id"].nunique()),
                                 "pretrained_encoder_may_have_seen_development_patients": True}
    train_rows = select_rows(frame, discovery, args.max_train_patients, "retention")
    if not usable:
        raise ValueError("no supported eICU reconstruction tasks")
    teacher = assets.model.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for arm in ARMS[1:]:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = copy.deepcopy(teacher)
        configure_training(model, arm)
        encoder = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("reconstruction_heads.")]
        heads = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("reconstruction_heads.")]
        groups = [{"params": heads, "lr": args.head_lr}]
        if encoder:
            groups.append({"params": encoder, "lr": args.encoder_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        arrays = explicit if arm == "C" else neutral
        log = []
        print(f"train {arm}: {len(usable)} targets; {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters", flush=True)
        for epoch in range(args.epochs):
            losses = []
            schedule = np.tile(np.arange(len(usable)), args.steps_per_target)
            rng.shuffle(schedule)
            for ti in schedule:
                target = usable[ti]
                idx = variables.index(target)
                pool = pools[target]
                rows, rr = paired_training_rows(rng, pool, train_rows, args.batch_size)
                inputs = batch_inputs(arrays, rows, hidden[target], explicit=arm == "C")
                label = torch.from_numpy(neutral["current"][rows, idx]).float()
                optimizer.zero_grad(set_to_none=True)
                if arm == "B":
                    with torch.no_grad():
                        state = model.encode_context_components(*inputs)
                else:
                    state = model.encode_context_components(*inputs)
                rec = model.reconstruction_heads[idx](state[0]).flatten()
                reconstruction = F.l1_loss(rec, label)
                preservation = reconstruction.new_zeros(())
                if arm != "B":
                    with torch.no_grad():
                        original = teacher.encode_context_components(*batch_inputs(neutral, rr))
                    updated = model.encode_context_components(*batch_inputs(arrays, rr))
                    preservation = components_loss(updated, original)
                loss = reconstruction + args.preservation_weight * preservation
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"nonfinite {arm} training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                optimizer.step()
                losses.append((float(reconstruction.detach()), float(preservation.detach())))
            item = {"epoch": epoch + 1, "reconstruction_l1_normalized": float(np.mean(losses, axis=0)[0]),
                    "unmasked_component_distillation": float(np.mean(losses, axis=0)[1])}
            log.append(item)
            print(arm, item, flush=True)
        # No future target encoder is trained here: this is a reconstruction
        # intervention on a JEPA release, not a new future-JEPA objective.
        original_state = teacher.state_dict()
        changed = [k for k, v in model.state_dict().items() if not torch.equal(v, original_state[k])]
        if arm == "B" and any(not k.startswith("reconstruction_heads.") for k in changed):
            raise AssertionError("readout-only arm changed forecast parameters")
        path = args.output_dir / f"{arm}.pt"
        torch.save({"state_dict": model.state_dict(), "arm": arm,
                    "base_sha256": report["release_binding"]["whole_body_joint_jepa.pt"],
                    "mask_mode": "explicit" if arm == "C" else "neutralized",
                    "reconstruction_latent": "anchor_context", "promotion_status": "candidate_only"}, path)
        report["training"][arm] = {"losses": log, "checkpoint": str(path), "sha256": sha256(path),
                                   "changed_tensors": len(changed), "future_heads_frozen": True}
        write_report(report, args)
        del model, optimizer


def load_arms(args, assets, report):
    result = {"A": assets}
    for arm in ARMS[1:]:
        path = args.output_dir / f"{arm}.pt"
        if sha256(path) != report["training"][arm]["sha256"]:
            raise ValueError("candidate checkpoint changed")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt["base_sha256"] != report["release_binding"]["whole_body_joint_jepa.pt"]:
            raise ValueError("candidate initialized from another release")
        model = copy.deepcopy(assets.model)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        result[arm] = replace(assets, model=model)
    return result


def latent_audit(model, arrays, rows, hidden, explicit):
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            inputs = batch_inputs(arrays, rows[start:start + 256], hidden, explicit=explicit)
            values.append(model.encode_context_components(*inputs)[0].numpy())
    x = np.concatenate(values).astype(float)
    singular = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    p = singular ** 2 / max(float(np.sum(singular ** 2)), 1e-12)
    rank = np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return {"effective_rank_energy_entropy": float(rank), "latent_dimensions": x.shape[1],
            "mean_coordinate_std": float(x.std(axis=0).mean())}


def evaluate_current(args, assets, arms, targets, report):
    for database, path, grid in (("eicu", args.eicu, True), ("mimic", args.mimic_current, False),
                                 ("nwicu", args.nwicu_current, False)):
        frame = read_examples(path, assets, database, 3, grid=grid)
        neutral = candidate_inputs(frame, assets, 3)
        explicit = candidate_inputs(frame, assets, 3, explicit=True)
        variables = assets.checkpoint["variables"]
        for target in targets:
            idx = variables.index(target)
            valid = frame[f"{target}_t"].notna() & frame[f"{target}_age_hr"].between(0, 1)
            if database == "eicu":
                _, development = patient_split(frame["subject_id"], args.seed)
                valid &= development
            rows = select_rows(frame, valid, args.max_patients, target)
            item = {"patients": len(rows), "arms": {}, "comparisons": {}}
            report["reconstruction"].setdefault(database, {})[target] = item
            if len(rows) < 200:
                item["status"] = "insufficient_fresh_support"
                continue
            hidden = sorted(variables.index(v) for v in ({target} | related_siblings(target)) & set(variables))
            item["hidden_variables"] = [variables[i] for i in hidden]
            y = frame.iloc[rows][f"{target}_t"].to_numpy()
            predictions = {}
            for arm, candidate in arms.items():
                arrays = explicit if arm == "C" else neutral
                predictions[arm] = infer(candidate, masked_arrays(arrays, hidden, arm == "C"), rows, target,
                                         batch_size=args.eval_batch_size)["reconstruction"]
                item["arms"][arm] = score(frame.iloc[rows], y, predictions[arm], float(assets.scaler.medians[idx]),
                                           f"{database}:{target}:current")
                item["arms"][arm]["latent"] = latent_audit(candidate.model, arrays, rows[:512], hidden, arm == "C")
            for first, second in (("B", "A"), ("C0", "B"), ("C", "B"), ("C", "C0")):
                item["comparisons"][f"{first}_vs_{second}"] = paired_screen(
                    frame.iloc[rows], y, predictions[first], predictions[second], f"{database}:{target}:{first}:{second}")
            print(f"current {database} {target}: " + " ".join(
                f"{a}={v['mae']:.4g}({v['diagnostic_pass']})" for a, v in item["arms"].items()), flush=True)
            write_report(report, args)


def evaluate_forecasts(args, assets, arms, report):
    cells = assets.source_selector["cells"]
    for horizon in sorted({int(v["horizon_hours"]) for v in cells.values()}):
        for db in ("mimic", "nwicu"):
            path = Path(args.mimic_template.format(horizon=horizon)) if db == "mimic" else args.nwicu_future
            frame = read_examples(path, assets, db, horizon, grid=db == "nwicu")
            neutral = candidate_inputs(frame, assets, horizon)
            explicit = candidate_inputs(frame, assets, horizon, explicit=True)
            for cell, config in cells.items():
                if int(config["horizon_hours"]) != horizon:
                    continue
                target = config["target"]
                item = {"source": config["source_model"]}
                report["forecast_retention"].setdefault(db, {})[cell] = item
                if target == "hypoxemia_event":
                    item["status"] = "separate_interval_event_labels_required"
                    continue
                if f"future_{target}" not in frame:
                    item["status"] = "no_compatible_future_target"
                    continue
                valid = valid_future_rows(frame, target)
                # All head comparisons honor the packaged multi-source fit exclusions.
                if db == "mimic" and cell in assets.multisource_heads["heads"]:
                    name = cell.replace("@", "_")
                    cache = args.previous_cache / f"{name}_mimic.parquet"
                    meta_path = args.previous_cache / f"{name}.json"
                    if not cache.exists() or not meta_path.exists():
                        item["status"] = "training_exclusions_unavailable"
                        continue
                    meta = json.loads(meta_path.read_text())["key"]
                    if meta["binding"] != assets.source_selector["release_binding"] or meta["mimic_sha256"] != sha256(path):
                        raise ValueError("incompatible exclusion cache")
                    subjects = pd.read_parquet(cache, columns=["subject_id"])["subject_id"].astype(str)
                    subjects = subjects.str.replace("MIMIC:", "mimic:", regex=False)
                    valid &= frame["subject_id"].isin(set(subjects))
                rows = select_rows(frame, valid, args.max_patients, cell)
                item["patients"] = len(rows)
                if len(rows) < 200:
                    item["status"] = "insufficient_supported_patients"
                    continue
                chosen = frame.iloc[rows]
                y = chosen[f"future_{target}"].to_numpy()
                persistence = chosen[f"{target}_t"].to_numpy()
                item.update(status="evaluated", persistence_mae=float(np.abs(y - persistence).mean()), arms={})
                reference = None
                for arm, candidate in arms.items():
                    if config["source_model"] == PERSISTENCE:
                        pred = persistence.copy()
                    else:
                        all_sources = infer(candidate, explicit if arm == "C" else neutral, rows, target, cell,
                                            args.eval_batch_size)
                        pred = all_sources[config["source_model"]]
                    if arm == "A":
                        reference = pred
                    item["arms"][arm] = {"mae": float(np.abs(y - pred).mean()),
                                          "max_absolute_change_vs_A": float(np.max(np.abs(pred - reference))),
                                          "vs_A": paired_screen(chosen, y, pred, reference, f"retention:{db}:{cell}:{arm}")}
                if item["arms"]["B"]["max_absolute_change_vs_A"] != 0:
                    raise AssertionError("B changed ordinary forecasts")
                print(f"forecast {db} {cell}: " + " ".join(f"{a}={v['mae']:.4g}" for a, v in item["arms"].items()), flush=True)
                write_report(report, args)


def write_report(report, args):
    summary = {}
    for db, tasks in report["reconstruction"].items():
        summary[db] = {arm: [t for t, r in tasks.items() if r.get("arms", {}).get(arm, {}).get("diagnostic_pass")]
                       for arm in ARMS}
    report["summary"] = summary
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/masked_abc_paired_20260903"))
    p.add_argument("--eicu", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet"))
    p.add_argument("--mimic-current", type=Path, default=Path("/private/tmp/mimiciv_observation_transitions_3h_full_v31_treatment_unit_safe.parquet"))
    p.add_argument("--nwicu-current", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet"))
    p.add_argument("--mimic-template", default="/private/tmp/mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe.parquet")
    p.add_argument("--nwicu-future", type=Path, default=Path("/private/tmp/nwicu_event_examples_41_grid.parquet"))
    p.add_argument("--previous-cache", type=Path, default=Path("/private/tmp/unified_source_predictions_20260902"))
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--steps-per-target", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--max-train-patients", type=int, default=3000)
    p.add_argument("--max-patients", type=int, default=4000)
    p.add_argument("--head-lr", type=float, default=0.003)
    p.add_argument("--encoder-lr", type=float, default=0.0001)
    p.add_argument("--preservation-weight", type=float, default=10.)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    binding = {f.name: sha256(f) for f in args.model_dir.iterdir() if f.is_file()}
    targets = sorted({v["target"] for v in assets.source_selector["cells"].values()} - {"hypoxemia_event"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "evaluate":
        report = json.loads((args.output_dir / "report.json").read_text())
        if report["release_binding"] != binding:
            raise ValueError("release changed since training")
        for key in ("seed", "epochs", "steps_per_target", "batch_size", "head_lr", "encoder_lr", "preservation_weight"):
            if report["settings"][key] != getattr(args, key):
                raise ValueError(f"training setting changed: {key}")
    else:
        if (args.output_dir / "report.json").exists():
            raise ValueError("use a fresh output directory; never overwrite an experiment")
        report = {"schema": "whole_body_jepa.masked_abc.v1", "release_binding": binding,
                  "script_sha256": sha256(Path(__file__)), "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "training_input_sha256": sha256(args.eicu), "training": {}, "reconstruction": {}, "forecast_retention": {},
                  "complete": False, "promotion_allowed": False, "runtime_modified": False,
                  "causal_claim_allowed": False, "next_measurement_time_modeled": False,
                  "untouched_lockbox_claim": False,
                  "protocol": "One fixed training seed, seven overlapping patient evaluation splits and every supported careunit. Wrong-patient derangements are NOT separately trained placebos. Fresh truth age <=1h; all hidden-target/sibling history removed. No external fitting. No reused intervals."}
    input_paths = {args.eicu, args.mimic_current, args.nwicu_current, args.nwicu_future}
    input_paths.update(Path(args.mimic_template.format(horizon=int(v["horizon_hours"])))
                       for v in assets.source_selector["cells"].values())
    hashes = {str(path): sha256(path) for path in sorted(input_paths)}
    if "input_hashes" in report and report["input_hashes"] != hashes:
        raise ValueError("input files changed since training")
    report["input_hashes"] = hashes
    report["paired_training_minibatches"] = True
    if args.stage in {"all", "train"}:
        train(args, assets, targets, report)
    if args.stage in {"all", "evaluate"}:
        arms = load_arms(args, assets, report)
        evaluate_current(args, assets, arms, targets, report)
        evaluate_forecasts(args, assets, arms, report)
        report["complete"] = True
    if binding != {f.name: sha256(f) for f in args.model_dir.iterdir() if f.is_file()}:
        raise ValueError("release files changed during experiment")
    write_report(report, args)
    print(json.dumps({"complete": report["complete"], "summary": report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
