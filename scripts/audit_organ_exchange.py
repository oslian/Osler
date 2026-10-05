"""Patient/hospital-heldout pilot of local recurrent organ information exchange."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from osler_jepa.organ_schema import CANONICAL_ORGAN_VARIABLES
from scripts.audit_state_conditioned_kidney import TARGETS, read_cohort, valid_target, source_roles
from scripts.train_release_masked_abc import candidate_inputs, batch_inputs
from whole_body_jepa.input_support import learn_input_support, apply_input_support
from whole_body_jepa.multisource_intervals import (
    SEEDS, _bootstrap, conformal_radius, representative_rows, sha256, stable_hash,
)
from whole_body_jepa.organ_exchange import ARMS, ORGANS, OrganExchangeResidual, mask_to_owner
from whole_body_jepa.runtime import load_release
from whole_body_jepa.source_selection import BASE, HEAD, MULTI, PERSISTENCE, predict_head_result


def label_arrays(frame, max_age, *, strict):
    valid, elapsed = [], []
    for target in TARGETS:
        good = valid_target(frame, target, max_age).copy()
        time_key = f"future_delta_t_hr_{target}"
        quality_key = f"future_label_quality_{target}"
        if strict and (time_key not in frame or quality_key not in frame):
            raise ValueError(f"strict source labels missing: {target}")
        dt = (pd.to_numeric(frame[time_key], errors="coerce").to_numpy(float)
              if time_key in frame else np.full(len(frame), 6.))
        good &= np.isfinite(dt) & (dt > 0) & (dt <= 48)
        valid.append(good)
        # Missing labels never define an inference query or a loss contribution.
        elapsed.append(np.where(good, dt, 6.))
    return np.stack(valid, 1), np.stack(elapsed, 1).astype(np.float32)


def select_patients(frame, eligible, cap, salt):
    subjects = sorted(set(frame.loc[eligible, "subject_id"]), key=lambda p: stable_hash(salt, p))[:cap]
    return set(subjects)


def prepare_frame(path, assets, *, database, cap, hospital_cap=0, exclusion=None, support=None):
    frame = read_cohort(path, assets, database, 6)
    original_patients = frame.subject_id.nunique()
    if exclusion is not None:
        previous_subjects, previous_stays = exclusion
        blocked = frame._raw_subject_id.isin(previous_subjects) | frame._raw_stay_id.isin(previous_stays)
        blocked_subjects = set(frame.loc[blocked, "subject_id"])
        frame = frame.loc[~frame.subject_id.isin(blocked_subjects)].reset_index(drop=True)
    valid, _ = label_arrays(frame, assets.max_age_hours, strict=database == "eicu")
    if database == "eicu":
        roles = source_roles(frame)
        hospital = roles["hospital"]
        development = ~frame.subject_id.isin(set(frame.loc[hospital, "subject_id"])).to_numpy()
        selected = select_patients(frame, valid.any(1) & development, cap, "organ_development")
        heldout = select_patients(frame, valid.any(1) & hospital, hospital_cap, "organ_hospital")
        selected |= heldout
    else:
        heldout = set()
        selected = select_patients(frame, valid.any(1), cap, "organ_external")
    frame = frame.loc[frame.subject_id.isin(selected)].reset_index(drop=True).copy()
    frame["hospital_holdout"] = frame.subject_id.isin(heldout)
    valid, _ = label_arrays(frame, assets.max_age_hours, strict=database == "eicu")
    eligible = np.flatnonzero(valid.any(1))
    chosen = eligible[representative_rows(frame.iloc[eligible], 20260906)]
    chosen.sort()
    if support is None:
        # Support is fitted only on a fixed source discovery subset shared
        # by all splits; the feature vocabulary is never learned externally.
        discovery = ~frame.hospital_holdout & frame.subject_id.map(
            lambda p: stable_hash("organ_support", p) % 10 == 0)
        support = learn_input_support(frame.loc[discovery], assets.checkpoint["variables"],
                                      max_age_hours=assets.max_age_hours)
        support_patients = set(frame.loc[discovery, "subject_id"])
    else:
        support_patients = set()
    frame["support_discovery"] = frame.subject_id.isin(support_patients)
    frame = apply_input_support(frame, support)
    summary = {"original_patients": int(original_patients), "selected_patients": len(selected),
               "history_rows": len(frame), "anchors": len(chosen),
               "hospital_patients": len(heldout), "support_discovery_patients": len(support_patients),
               "strict_event_time_labels": database == "eicu",
               "pretraining_subjects_stays_excluded": exclusion is not None}
    return frame, chosen, support, summary


def router_predictions(output, current, assets, indexes):
    values = []
    for index, target in zip(indexes, TARGETS):
        cell = f"{target}@6h"
        decision = assets.source_selector["cells"].get(cell)
        baseline = current[:, index].numpy().copy()
        if decision is None or not decision["can_move"] or decision["source_model"] == PERSISTENCE:
            values.append(baseline)
            continue
        source = decision["source_model"]
        if source == BASE:
            result = output["value"][:, index].numpy()
        elif source in {HEAD, MULTI}:
            organ = int(assets.model.variable_module_ids[index])
            features = np.concatenate([output["context_latent"].numpy(), output["belief_state"].numpy(),
                output["predicted_target_latent"][:, index].numpy(), output["organ_latents"][:, organ].numpy(),
                output["value"][:, index:index + 1].numpy(), current.numpy(), current[:, index:index + 1].numpy()], 1)
            bundle = assets.heads if source == HEAD else assets.multisource_heads
            physical_current = baseline * assets.scaler.scales[index] + assets.scaler.medians[index]
            physical, _ = predict_head_result(bundle["heads"][cell], features, physical_current, runtime=source == HEAD)
            result = (physical - assets.scaler.medians[index]) / assets.scaler.scales[index]
        else:
            raise ValueError(f"unsupported reference source: {source}")
        values.append(result)
    return np.stack(values, 1)


def encode(frame, chosen, assets, *, strict):
    variables = assets.checkpoint["variables"]
    indexes = [variables.index(t) for t in TARGETS]
    arrays = candidate_inputs(frame, assets, 6)
    valid, elapsed = label_arrays(frame.iloc[chosen], assets.max_age_hours, strict=strict)
    current = arrays["current"][chosen][:, indexes]
    truth = np.stack([frame.iloc[chosen][f"future_{t}"].to_numpy(float) for t in TARGETS], 1)
    truth = (truth - assets.scaler.medians[indexes]) / assets.scaler.scales[indexes]
    truth = np.where(valid, truth, 0).astype(np.float32)
    observed = np.stack([np.isfinite(frame.iloc[chosen][f"{t}_t"])
        & frame.iloc[chosen][f"{t}_age_hr"].between(0, assets.max_age_hours) for t in TARGETS], 1)
    parts = {k: [] for k in ("organs", "presence", "base", "base_latent", "teacher", "router", "joint_nominal")}
    net = assets.model.eval()
    with torch.inference_mode():
        for start in range(0, len(chosen), 128):
            rows = chosen[start:start + 128]
            sl = slice(start, start + len(rows))
            batch = batch_inputs(arrays, rows)
            target_time = torch.full_like(batch[1], 6.)
            target_time[:, indexes] = torch.from_numpy(elapsed[sl])
            full = net(*batch, torch.full((len(rows),), 6.))
            parts["joint_nominal"].append(full["value"][:, indexes].numpy())
            parts["router"].append(router_predictions(full, batch[1], assets, indexes))
            organs, presence = [], []
            for organ in ORGANS:
                allowed = [variables.index(v) for v in CANONICAL_ORGAN_VARIABLES[organ] if v in variables]
                local = mask_to_owner(*batch, allowed=allowed)
                local_latent = net.encode_latent(*local)
                available = np.zeros(len(rows), dtype=bool)
                for variable in CANONICAL_ORGAN_VARIABLES[organ]:
                    if variable in variables:
                        available |= (np.isfinite(frame.iloc[rows][f"{variable}_t"])
                            & frame.iloc[rows][f"{variable}_age_hr"].between(0, assets.max_age_hours)).to_numpy()
                organs.append(local_latent.numpy())
                presence.append(available)
                if organ == "kidneys":
                    owner = net(*local, torch.full((len(rows),), 6.), target_horizon=target_time)
                    parts["base"].append(owner["value"][:, indexes].numpy())
                    parts["base_latent"].append(owner["predicted_target_latent"][:, indexes].numpy())
            parts["organs"].append(np.stack(organs, 1))
            parts["presence"].append(np.stack(presence, 1).astype(np.float32))
            delta = torch.zeros_like(batch[1])
            mask = torch.zeros_like(batch[1])
            delta[:, indexes] = torch.from_numpy(truth[sl] - current[sl])
            mask[:, indexes] = torch.from_numpy(valid[sl].astype(np.float32))
            teacher, _ = net._encode_target_specific_values(delta * mask, mask)
            parts["teacher"].append(teacher[:, indexes].numpy())
    result = {key: torch.from_numpy(np.concatenate(value).astype(np.float32)) for key, value in parts.items()}
    result.update(current=torch.from_numpy(current), current_mask=torch.from_numpy(observed.astype(np.float32)),
                  hours=torch.from_numpy(elapsed), truth=torch.from_numpy(truth),
                  valid=torch.from_numpy(valid.astype(np.float32)))
    return result


def run_model(model, data, rows):
    return model(*(data[key][rows] for key in
        ("organs", "presence", "current", "current_mask", "hours", "base", "base_latent")))


def train(data, rows, *, arm, seed, epochs):
    torch.manual_seed(20260906)  # initialization separate from patient split
    model = OrganExchangeResidual([ORGANS.index("kidneys")] * len(TARGETS), arm=arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    log = []
    for _ in range(epochs):
        losses = []
        permutation = rng.permutation(rows)
        for start in range(0, len(rows), 128):
            batch = permutation[start:start + 128]
            out = run_model(model, data, batch)
            mask = data["valid"][batch]
            count = mask.sum(0)
            supported = count > 0
            numeric = F.smooth_l1_loss(out["value"], data["truth"][batch], reduction="none")
            latent = (out["future_latent"] - data["teacher"][batch]).square().mean(-1)
            loss = (((numeric + .5 * latent) * mask).sum(0) / count.clamp_min(1))[supported].mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        log.append(float(np.mean(losses)))
    return model.eval(), log


def predictions(model, data):
    with torch.inference_mode():
        return torch.cat([run_model(model, data, np.arange(start, min(start + 128, len(data["current"]))))["value"]
                          for start in range(0, len(data["current"]), 128)]).numpy()


def evaluate(meta, data, pred, rows, radii, scales, salt):
    report = {}
    truth, current = data["truth"].numpy(), data["current"].numpy()
    values = dict(pred, owner=data["base"].numpy(), persistence=current, router=data["router"].numpy())
    for i, target in enumerate(TARGETS):
        selected = rows[data["valid"][rows, i].numpy() > 0]
        n = len(selected)
        if n < 30:
            report[target] = {"patients": n, "status": "insufficient_support", "point_screen_pass": False}
            continue
        errors = {arm: np.abs(v[selected, i] - truth[selected, i]) * scales[i] for arm, v in values.items()}
        comparisons = {}
        units = meta.iloc[selected].careunit.to_numpy()
        for arm in values:
            if arm == "sparse":
                continue
            delta = errors["sparse"] - errors[arm]
            pooled = _bootstrap(delta, stable_hash(salt, target, arm))
            per_unit = {str(u): {"patients": int((units == u).sum()),
                                **_bootstrap(delta[units == u], stable_hash(salt, target, arm, u))}
                        for u in sorted(set(units)) if (units == u).sum() >= 30 and u != "unknown"}
            comparisons[arm] = {"pooled": pooled, "careunits": per_unit}
        uncertainty = {}
        for arm in ARMS:
            radius = radii[arm][i]
            if radius is None:
                uncertainty[arm] = {"status": "insufficient_calibration", "pass": False}
                continue
            hit = np.abs(values[arm][selected, i] - truth[selected, i]) <= radius
            unit_coverage = {str(u): float(hit[units == u].mean()) for u in sorted(set(units))
                             if (units == u).sum() >= 30 and u != "unknown"}
            uncertainty[arm] = {"coverage": float(hit.mean()), "width": float(2 * radius * scales[i]),
                "careunit_coverage": unit_coverage,
                "pass": bool(n >= 100 and .87 <= hit.mean() <= .93 and len(unit_coverage) >= 2
                             and all(.87 <= x <= .93 for x in unit_coverage.values()))}
        report[target] = {"patients": n, "mae": {arm: float(err.mean()) for arm, err in errors.items()},
            "comparisons": comparisons, "intervals": uncertainty,
            "point_screen_pass": bool(n >= 100 and all(c["pooled"]["pass"] for c in comparisons.values())),
            "diagnostic_joint_nominal_mae": float(np.abs(data["joint_nominal"].numpy()[selected, i]
                                                        - truth[selected, i]).mean() * scales[i]),
            "diagnostic_residual_saturation": {arm: float((np.abs(
                pred[arm][selected, i] - data["base"].numpy()[selected, i]) >= .2375).mean()) for arm in ARMS},
            "actual_delta_t_quantiles": np.quantile(data["hours"][selected, i].numpy(), [0, .5, 1]).tolist()}
    return report


def run(args):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    assets = load_release(args.model_dir)
    binding = sha256(args.model_dir / "whole_body_joint_jepa.pt")
    training_file = Path(assets.checkpoint["training"]["event_examples"])
    old = pd.read_parquet(training_file, columns=["subject_id", "stay_id"])
    exclusion = (set(old.subject_id.astype(str)), set(old.stay_id.astype(str)))
    print("prepare excluded, patient-bounded eICU cohort", flush=True)
    source, rows, support, source_summary = prepare_frame(args.eicu, assets, database="eicu",
        cap=args.source_patients, hospital_cap=args.hospital_patients, exclusion=exclusion)
    print(json.dumps(source_summary), flush=True)
    print("encode owner-restricted source histories with the same JEPA", flush=True)
    data = encode(source, rows, assets, strict=True)
    meta = source.iloc[rows].reset_index(drop=True)
    del source
    print("prepare supplementary MIMIC nominal-time validation", flush=True)
    external, external_rows, _, external_summary = prepare_frame(args.mimic, assets, database="mimic",
        cap=args.external_patients, support=support)
    print(json.dumps(external_summary), flush=True)
    ext = encode(external, external_rows, assets, strict=False)
    ext_meta = external.iloc[external_rows].reset_index(drop=True)
    del external
    # Seven actual train/calibration/test partitions, not seven re-samples of
    # one fitted model's test predictions. Support discovery is always train-only.
    report = {"schema": "local_organ_exchange_pilot.v1", "protocol_sha256": sha256(Path("ORGAN_EXCHANGE_PROTOCOL_20260906.md")),
        "checkpoint_sha256": binding, "source": source_summary, "external": external_summary,
        "source_support": support, "settings": {"epochs": args.epochs, "split_seeds": list(SEEDS),
            "initialization_seed": 20260906, "latent": 64, "learning_rate": .001,
            "numeric_loss": "masked_target_balanced_smooth_l1", "target_JEPA_weight": .5,
            "frozen_owner_encoder": True, "fixed_integer_rollout_interpolated_at_query_time": True},
        "runtime_modified": False, "promotion_allowed": False, "causal_claim_allowed": False,
        "router_reference_status": {t: {
            "catalog_cell_exists": f"{t}@6h" in assets.source_selector["cells"],
            "decision": assets.source_selector["cells"].get(f"{t}@6h"),
            "missing_cell_policy": "persistence_reference_only_not_a_shipped_cell"} for t in TARGETS},
        "independent_external_lockbox": False, "results": []}
    scales = assets.scaler.scales[[assets.checkpoint["variables"].index(t) for t in TARGETS]]
    hospital = meta.hospital_holdout.to_numpy()
    fixed_train = meta.support_discovery.to_numpy()
    for seed in SEEDS:
        bucket = np.array([stable_hash("organ_exchange_split", seed, s) % 10 for s in meta.subject_id])
        roles = {"train": np.flatnonzero(~hospital & ((bucket < 6) | fixed_train)),
                 "calibration": np.flatnonzero(~hospital & ~fixed_train & (bucket >= 6) & (bucket < 8)),
                 "patient_test": np.flatnonzero(~hospital & ~fixed_train & (bucket >= 8)),
                 "hospital": np.flatnonzero(hospital)}
        sets = [set(meta.iloc[v].subject_id) for v in roles.values()]
        if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
            raise ValueError("overlapping patient partitions")
        item = {"seed": seed, "role_patients": {k: len(v) for k, v in roles.items()}, "training": {}}
        pred, ext_pred, radii = {}, {}, {}
        for arm in ARMS:
            model, loss = train(data, roles["train"], arm=arm, seed=seed, epochs=args.epochs)
            pred[arm], ext_pred[arm] = predictions(model, data), predictions(model, ext)
            radii[arm] = []
            for i in range(len(TARGETS)):
                valid = roles["calibration"][data["valid"][roles["calibration"], i].numpy() > 0]
                radius = (conformal_radius(np.abs(pred[arm][valid, i] - data["truth"][valid, i].numpy()))
                          if len(valid) >= 30 else None)
                radii[arm].append(radius)
            item["training"][arm] = {"loss": loss, "parameters": sum(p.numel() for p in model.parameters()),
                "calibration_radii_normalized": radii[arm], "adjacency": model.state_layer.adjacency.tolist()}
            torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed,
                        "owners": [3] * len(TARGETS), "targets": TARGETS, "organs": ORGANS,
                        "source_support": support, "checkpoint_sha256": binding,
                        "promotion_allowed": False}, args.output / f"candidate_{seed}_{arm}.pt")
            print(json.dumps({"seed": seed, "arm": arm, "final_loss": loss[-1]}), flush=True)
        item["evaluation"] = {role: evaluate(meta, data, pred, roles[role], radii, scales, f"{seed}:{role}")
                               for role in ("patient_test", "hospital")}
        item["evaluation"]["mimic_nominal"] = evaluate(ext_meta, ext, ext_pred, np.arange(len(ext_meta)),
                                                       radii, scales, f"{seed}:mimic")
        report["results"].append(item)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({"seed": seed, "source_point_pass": [t for t, r in item["evaluation"]["patient_test"].items()
                                                                 if r["point_screen_pass"]]}), flush=True)
    report["all_seed_point_screen_targets"] = [t for t in TARGETS if all(
        result["evaluation"][role][t]["point_screen_pass"] for result in report["results"]
        for role in ("patient_test", "hospital", "mimic_nominal"))]
    if sha256(args.model_dir / "whole_body_joint_jepa.pt") != binding:
        raise ValueError("release checkpoint changed during the audit")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("FINISHED " + json.dumps(report["all_seed_point_screen_targets"]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--output", type=Path, default=Path("artifacts/organ_exchange_20260906"))
    p.add_argument("--eicu", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_future_label_repaired/eicu_aki_transitions_6h.parquet"))
    p.add_argument("--mimic", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/mimiciv_observation_transitions_6h_full_v31_treatment_unit_safe.parquet"))
    p.add_argument("--source-patients", type=int, default=1800)
    p.add_argument("--hospital-patients", type=int, default=600)
    p.add_argument("--external-patients", type=int, default=800)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
