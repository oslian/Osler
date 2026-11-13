"""Test persistence as a learned stable-state expert on kidney-related targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

from osler_jepa.shared_release_features import FEATURE_SLICES
from scripts.audit_release_masked_cells import donor_indices
from scripts.train_release_masked_abc import candidate_inputs
from whole_body_jepa.multisource_intervals import (
    BOUNDS,
    SEEDS,
    _bootstrap,
    conformal_radius,
    patient_split,
    representative_rows,
    sha256,
    stable_hash,
)
from whole_body_jepa.runtime import load_release
from whole_body_jepa.preanchor_context import precise_treatment_source_columns
from whole_body_jepa.state_conditioned import (
    DYNAMIC_THRESHOLDS,
    empirical_crps,
    fit_state_conditioned_head,
    mixture_samples,
    predict_state_conditioned,
)


TARGETS = ("creatinine", "bun", "urine_output", "potassium", "bicarbonate")
ROLE_CAPS = {"train": 12000, "tune": 5000, "test": 5000, "hospital": 5000, "external": 12000}


def read_cohort(path: Path, assets, database: str, horizon: int) -> pd.DataFrame:
    names = set(pq.ParquetFile(path).schema_arrow.names)
    metadata = {
        "subject_id", "stay_id", "t", "anchor_time", "hours_since_onset",
        "onset_hour", "hadm_id", "careunit", "first_careunit", "unittype",
        "hospitalid", "horizon_hours",
    }
    context_actions = ("fluids", "vasopressor", "diuretics", "renal_replacement", "nephrotoxin")
    wanted_context = {
        f"hist_{action}{suffix}"
        for action in context_actions
        for suffix in ("", "_evidence_count", "_dose_observed")
    }
    wanted = metadata | {
        f"{variable}{suffix}"
        for variable in assets.checkpoint["variables"]
        for suffix in ("_t", "_age_hr")
    }
    for target in TARGETS:
        wanted |= {
            f"{target}_tp{horizon}", f"future_{target}",
            f"future_label_quality_{target}", f"future_label_weight_{target}",
            f"future_delta_t_hr_{target}",
        }
    frame = pd.read_parquet(
        path,
        columns=sorted(names & (wanted | wanted_context | set(precise_treatment_source_columns()))),
    )
    if "horizon_hours" in frame:
        frame = frame.loc[pd.to_numeric(frame["horizon_hours"], errors="coerce").eq(float(horizon))]
    frame["_raw_subject_id"] = frame["subject_id"].astype(str)
    frame["_raw_stay_id"] = frame["stay_id"].astype(str)
    frame["subject_id"] = database + ":" + frame["_raw_subject_id"]
    frame["stay_id"] = database + ":" + frame["_raw_stay_id"]
    frame["t"] = pd.to_datetime(frame["t"] if "t" in frame else frame["anchor_time"], errors="coerce")
    frame = frame.loc[frame["t"].notna()].copy()
    unit = next((name for name in ("first_careunit", "careunit", "unittype") if name in frame), None)
    frame["careunit"] = frame[unit].fillna("unknown").astype(str) if unit else "unknown"
    frame["hospitalid"] = frame.get("hospitalid", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    frame = frame.sort_values(["stay_id", "t"], kind="stable").drop_duplicates(["stay_id", "t"]).reset_index(drop=True)
    origin = frame.groupby("stay_id")["t"].transform("min")
    frame["_time_hours"] = (frame["t"] - origin).dt.total_seconds() / 3600.0
    frame["hours_since_onset"] = pd.to_numeric(
        frame.get("hours_since_onset", frame["_time_hours"]), errors="coerce"
    ).fillna(frame["_time_hours"])
    for variable in assets.checkpoint["variables"]:
        for suffix in ("_t", "_age_hr"):
            column = f"{variable}{suffix}"
            frame[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
    for target in TARGETS:
        legacy = f"{target}_tp{horizon}"
        if legacy in frame:
            frame[f"future_{target}"] = pd.to_numeric(frame[legacy], errors="coerce")
        elif f"future_{target}" in frame:
            frame[f"future_{target}"] = pd.to_numeric(frame[f"future_{target}"], errors="coerce")
        else:
            frame[f"future_{target}"] = np.nan
    return frame


def valid_target(frame: pd.DataFrame, target: str, max_age: float) -> np.ndarray:
    valid = (
        np.isfinite(frame[f"{target}_t"])
        & np.isfinite(frame[f"future_{target}"])
        & frame[f"{target}_age_hr"].between(0.0, max_age)
    )
    quality = f"future_label_quality_{target}"
    if quality in frame:
        valid &= frame[quality].astype(str).str.lower().isin({"exact", "strict"})
    weight = f"future_label_weight_{target}"
    if weight in frame:
        valid &= pd.to_numeric(frame[weight], errors="coerce").fillna(0.0) > 0.0
    actual = f"future_delta_t_hr_{target}"
    if actual in frame:
        valid &= pd.to_numeric(frame[actual], errors="coerce") > 0.0
    return valid.to_numpy()


def source_roles(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    hospital_holdout = np.asarray([
        stable_hash("state_conditioned_hospital", value) % 5 == 0
        for value in frame["hospitalid"]
    ])
    # A patient seen at a held-out hospital cannot contribute development rows elsewhere.
    heldout_patients = set(frame.loc[hospital_holdout, "subject_id"])
    development = ~frame["subject_id"].isin(heldout_patients).to_numpy()
    bucket = np.asarray([
        stable_hash("state_conditioned_patient", value) % 10
        for value in frame["subject_id"]
    ])
    return {
        "train": development & (bucket < 6),
        "tune": development & (bucket >= 6) & (bucket < 8),
        "test": development & (bucket >= 8),
        "hospital": hospital_holdout,
    }


def representative_target_rows(
    frame: pd.DataFrame, valid: np.ndarray, role: np.ndarray, cap: int, salt: str
) -> np.ndarray:
    eligible = np.flatnonzero(valid & role)
    if not len(eligible):
        return eligible
    selected = representative_rows(frame.iloc[eligible], stable_hash("state_conditioned_anchor", salt))
    rows = eligible[selected]
    hashes = np.asarray([stable_hash("state_conditioned_cap", salt, value) for value in frame.iloc[rows]["subject_id"]])
    return rows[np.argsort(hashes, kind="stable")[:cap]]


def encode_features(frame: pd.DataFrame, rows: np.ndarray, assets, horizon: int) -> dict[str, np.ndarray]:
    arrays = candidate_inputs(frame, assets, horizon)
    rows = np.asarray(sorted(set(map(int, rows))), dtype=np.int64)
    feature_parts = {target: [] for target in TARGETS}
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            current = torch.from_numpy(arrays["current"][batch]).float()
            output = assets.model(
                torch.from_numpy(arrays["input"][arrays["history"][batch]]).float(),
                current,
                torch.from_numpy(arrays["mask"][batch]).float(),
                torch.from_numpy(arrays["ages"][batch]).float(),
                torch.from_numpy(arrays["presence"][batch]).float(),
                torch.from_numpy(arrays["horizon"][batch]).float(),
            )
            context = output["context_latent"].cpu().numpy()
            belief = output["belief_state"].cpu().numpy()
            target_future = output["predicted_target_latent"].cpu().numpy()
            organs = output["organ_latents"].cpu().numpy()
            base = output["value"].cpu().numpy()
            measurement = arrays["current"][batch]
            for target in TARGETS:
                index = assets.checkpoint["variables"].index(target)
                organ = int(assets.model.variable_module_ids[index])
                feature_parts[target].append(np.concatenate(
                    [context, belief, target_future[:, index], organs[:, organ], base[:, index, None],
                     measurement, measurement[:, index, None]], axis=1
                ).astype(np.float32))
    return {target: np.concatenate(parts) for target, parts in feature_parts.items()}


def feature_index(rows: np.ndarray) -> dict[int, int]:
    return {int(row): index for index, row in enumerate(rows)}


def positions(rows: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.asarray([mapping[int(row)] for row in rows], dtype=np.int64)


def patient_bootstrap(frame: pd.DataFrame, values: np.ndarray, salt: str) -> dict:
    grouped = pd.Series(values, index=frame["subject_id"].to_numpy()).groupby(level=0).mean().to_numpy()
    return _bootstrap(grouped, stable_hash("state_conditioned", salt))


def comparison_gate(frame: pd.DataFrame, candidate: np.ndarray, reference: np.ndarray, truth: np.ndarray, salt: str) -> dict:
    delta = np.abs(candidate - truth) - np.abs(reference - truth)
    splits = []
    for seed in SEEDS:
        _, heldout = patient_split(frame["subject_id"], seed)
        splits.append({"seed": seed, **patient_bootstrap(frame.loc[heldout], delta[heldout], f"{salt}:{seed}")})
    unit_patient = pd.DataFrame({
        "unit": frame["careunit"].astype(str).to_numpy(),
        "patient": frame["subject_id"].to_numpy(),
        "delta": delta,
    }).groupby(["unit", "patient"])["delta"].mean().groupby(level=0).mean()
    unit = _bootstrap(unit_patient.to_numpy(), stable_hash("state_conditioned", salt, "unit"))
    late_cutoff = frame["_time_hours"].quantile(0.8)
    late = frame["_time_hours"].to_numpy() >= late_cutoff
    time = patient_bootstrap(frame.loc[late], delta[late], f"{salt}:late")
    return {
        "all": patient_bootstrap(frame, delta, f"{salt}:all"),
        "patient_splits": splits,
        "careunit": unit,
        "late_stay": time,
        "pass": all(item["pass"] for item in splits) and unit["pass"] and time["pass"],
    }


def coverage_gate(frame: pd.DataFrame, truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    hit = (truth >= lower) & (truth <= upper)
    patient = pd.Series(hit, index=frame["subject_id"].to_numpy()).groupby(level=0).mean()
    unit = pd.DataFrame({"unit": frame["careunit"].astype(str), "patient": frame["subject_id"], "hit": hit})
    unit = unit.groupby(["unit", "patient"])["hit"].mean().groupby(level=0).mean()
    values = {
        "row": float(hit.mean()),
        "patient_equal": float(patient.mean()),
        "careunit_equal": float(unit.mean()),
        "mean_width": float(np.mean(upper - lower)),
        "median_width": float(np.median(upper - lower)),
    }
    values["pass"] = all(BOUNDS[0] <= values[key] <= BOUNDS[1] for key in ("row", "patient_equal", "careunit_equal"))
    return values


def evaluate(
    frame: pd.DataFrame,
    rows: np.ndarray,
    features: np.ndarray,
    target: str,
    head: dict,
    placebo: dict,
    cutoff: float,
    interval_scale: float,
    role: str,
) -> dict:
    selected = frame.iloc[rows].reset_index(drop=True)
    current = selected[f"{target}_t"].to_numpy(dtype=float)
    truth = selected[f"future_{target}"].to_numpy(dtype=float)
    delta = truth - current
    shared = features[:, :FEATURE_SLICES["patient_belief"].stop]
    actual = predict_state_conditioned(head, shared, features, cutoff=cutoff)
    placebo_prediction = predict_state_conditioned(placebo, shared, features, cutoff=cutoff)
    point = current + actual["point_delta"]
    plain = current + actual["plain_delta"]
    placebo_point = current + placebo_prediction["point_delta"]
    samples = mixture_samples(
        actual["dynamic_probability"], actual["dynamic_quantiles"], head["stable_delta_quantiles"]
    )
    persistence_samples = np.tile(head["persistence_delta_quantiles"], (len(rows), 1))
    candidate_crps = empirical_crps(samples, delta)
    persistence_crps = empirical_crps(persistence_samples, delta)
    center = current + np.quantile(samples, 0.50, axis=1)
    base_lower = current + np.quantile(samples, 0.05, axis=1)
    base_upper = current + np.quantile(samples, 0.95, axis=1)
    lower = center - interval_scale * (center - base_lower)
    upper = center + interval_scale * (base_upper - center)
    dynamic = np.abs(delta) >= float(head["dynamic_threshold"])
    auc = None
    if dynamic.any() and (~dynamic).any():
        auc = float(roc_auc_score(dynamic, actual["dynamic_probability"]))
    comparisons = {
        "persistence": comparison_gate(selected, point, current, truth, f"{target}:{role}:persistence"),
        "plain_equal_input_head": comparison_gate(selected, point, plain, truth, f"{target}:{role}:plain"),
        "capacity_matched_placebo": comparison_gate(selected, point, placebo_point, truth, f"{target}:{role}:placebo"),
    }
    crps_delta = candidate_crps - persistence_crps
    crps_gate = patient_bootstrap(selected, crps_delta, f"{target}:{role}:crps")
    result = {
        "rows": int(len(selected)),
        "patients": int(selected["subject_id"].nunique()),
        "dynamic_fraction": float(dynamic.mean()),
        "dynamic_auc": auc,
        "mae": {
            "state_conditioned": float(np.mean(np.abs(point - truth))),
            "persistence": float(np.mean(np.abs(current - truth))),
            "plain_equal_input_head": float(np.mean(np.abs(plain - truth))),
            "capacity_matched_placebo": float(np.mean(np.abs(placebo_point - truth))),
            "stable_state_conditioned": float(np.mean(np.abs(point[~dynamic] - truth[~dynamic]))) if (~dynamic).any() else None,
            "stable_persistence": float(np.mean(np.abs(current[~dynamic] - truth[~dynamic]))) if (~dynamic).any() else None,
            "dynamic_state_conditioned": float(np.mean(np.abs(point[dynamic] - truth[dynamic]))) if dynamic.any() else None,
            "dynamic_persistence": float(np.mean(np.abs(current[dynamic] - truth[dynamic]))) if dynamic.any() else None,
        },
        "crps": {
            "state_conditioned": float(candidate_crps.mean()),
            "empirical_persistence": float(persistence_crps.mean()),
            "paired": crps_gate,
        },
        "comparisons": comparisons,
        "coverage": coverage_gate(selected, truth, lower, upper),
    }
    result["full_gate_pass"] = bool(
        all(item["pass"] for item in comparisons.values())
        and crps_gate["pass"]
        and result["coverage"]["pass"]
    )
    return result


def tune_cutoff(frame, rows, features, target, head) -> tuple[float, float]:
    selected = frame.iloc[rows].reset_index(drop=True)
    current = selected[f"{target}_t"].to_numpy(dtype=float)
    truth = selected[f"future_{target}"].to_numpy(dtype=float)
    shared = features[:, :FEATURE_SLICES["patient_belief"].stop]
    choices = []
    delta = truth - current
    dynamic = np.abs(delta) >= float(head["dynamic_threshold"])
    for cutoff in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99):
        prediction = predict_state_conditioned(head, shared, features, cutoff=float(cutoff))
        error = np.abs(current + prediction["point_delta"] - truth)
        patient_mae = pd.Series(error, index=selected["subject_id"]).groupby(level=0).mean().mean()
        stable_harm = float(error[~dynamic].mean() - np.abs(delta[~dynamic]).mean())
        dynamic_gain = float(error[dynamic].mean() - np.abs(delta[dynamic]).mean())
        eligible = stable_harm <= 0.02 * float(head["dynamic_threshold"]) and dynamic_gain < 0.0
        choices.append((not eligible, float(patient_mae), float(cutoff)))
    cutoff = min(choices)[2]
    prediction = predict_state_conditioned(head, shared, features, cutoff=cutoff)
    samples = mixture_samples(
        prediction["dynamic_probability"], prediction["dynamic_quantiles"], head["stable_delta_quantiles"]
    )
    center = current + np.quantile(samples, 0.50, axis=1)
    low = current + np.quantile(samples, 0.05, axis=1)
    high = current + np.quantile(samples, 0.95, axis=1)
    lower_width = np.maximum(center - low, 1e-6)
    upper_width = np.maximum(high - center, 1e-6)
    nonconformity = np.maximum((center - truth) / lower_width, (truth - center) / upper_width)
    interval_scale = conformal_radius(np.maximum(nonconformity, 0.0))
    return cutoff, interval_scale


def prepare_source(args, assets):
    frame = read_cohort(args.eicu, assets, "eicu", args.horizon)
    roles = source_roles(frame)
    selected = {}
    for target in TARGETS:
        valid = valid_target(frame, target, assets.max_age_hours)
        selected[target] = {
            role: representative_target_rows(frame, valid, mask, ROLE_CAPS[role], f"{target}:{role}")
            for role, mask in roles.items()
        }
    union = np.asarray(sorted({int(row) for target in selected.values() for rows in target.values() for row in rows}))
    features = encode_features(frame, union, assets, args.horizon)
    lookup = feature_index(union)
    return frame, selected, features, lookup


def prepare_external(args, assets):
    frame = read_cohort(args.mimic, assets, "mimic", args.horizon)
    selected = {}
    all_rows = np.ones(len(frame), dtype=bool)
    for target in TARGETS:
        valid = valid_target(frame, target, assets.max_age_hours)
        selected[target] = representative_target_rows(
            frame, valid, all_rows, ROLE_CAPS["external"], f"{target}:external"
        )
    union = np.asarray(sorted({int(row) for rows in selected.values() for row in rows}))
    features = encode_features(frame, union, assets, args.horizon)
    return frame, selected, features, feature_index(union)


def write_findings(report: dict, path: Path) -> None:
    lines = [
        "# State-Conditioned Kidney Forecast Audit", "", "## Result", "",
        "The frozen shared whole-body JEPA was not retrained. Each target uses the same",
        "legal pre-anchor representation. A classifier estimates `P(dynamic)` from shared",
        "body context and belief state; stable predictions stay at persistence, while the",
        "dynamic expert predicts signed delta quantiles. A plain equal-input head and a",
        "capacity-matched shuffled-state model are mandatory controls.", "",
        "| Target | Cutoff | eICU patient | eICU hospital | MIMIC-IV | Promotion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target, item in report["targets"].items():
        lines.append(
            f"| {target}@{report['horizon_hours']}h | {item['cutoff']:.2f} | "
            f"{item['evaluation']['eicu_patient']['full_gate_pass']} | "
            f"{item['evaluation']['eicu_hospital']['full_gate_pass']} | "
            f"{item['evaluation']['mimic_external']['full_gate_pass']} | "
            f"{item['promotion_allowed']} |"
        )
    lines += [
        "", "## Boundaries", "",
        "This is a factual development/transfer audit. MIMIC-IV has been inspected before",
        "and is not described as an untouched lockbox. No external labels selected a model",
        "or cutoff. A target can be promoted only if point MAE beats persistence, the plain",
        "equal-input head and shuffled-state placebo across all patient/domain/time gates;",
        "CRPS beats the empirical persistence distribution; and 90% interval coverage is",
        "within [0.87, 0.93]. Failure leaves the current release and persistence fallback unchanged.",
        "Measurement-time prediction is a separate task and was not modeled here. No causal",
        "or clinical decision claim is allowed.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    assets = load_release(args.model_dir)
    report = {
        "schema": "whole_body_jepa.state_conditioned_kidney_audit.v1",
        "horizon_hours": args.horizon,
        "release_binding": {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()},
        "source_database": "eICU",
        "external_database": "MIMIC-IV",
        "external_is_untouched_lockbox": False,
        "dynamic_thresholds": DYNAMIC_THRESHOLDS,
        "coverage_bounds": list(BOUNDS),
        "targets": {},
        "promotion_allowed": False,
        "runtime_modified": False,
        "causal_claim_allowed": False,
        "next_measurement_time_modeled": False,
    }
    print("prepare eICU source features", flush=True)
    source_frame, source_rows, source_features, source_lookup = prepare_source(args, assets)
    heads = {}
    for target in TARGETS:
        print(f"fit state-conditioned head: {target}", flush=True)
        rows = source_rows[target]
        train_pos = positions(rows["train"], source_lookup)
        tune_pos = positions(rows["tune"], source_lookup)
        train_feature = source_features[target][train_pos]
        tune_feature = source_features[target][tune_pos]
        train_selected = source_frame.iloc[rows["train"]]
        delta = (
            train_selected[f"future_{target}"].to_numpy(dtype=float)
            - train_selected[f"{target}_t"].to_numpy(dtype=float)
        )
        shared = train_feature[:, :FEATURE_SLICES["patient_belief"].stop]
        head = fit_state_conditioned_head(
            shared, train_feature, delta,
            dynamic_threshold=DYNAMIC_THRESHOLDS[target], seed=args.seed,
        )
        donor = donor_indices(train_selected["subject_id"], args.seed)
        placebo = fit_state_conditioned_head(
            shared[donor], train_feature[donor], delta,
            dynamic_threshold=DYNAMIC_THRESHOLDS[target], seed=args.seed,
        )
        cutoff, interval_scale = tune_cutoff(
            source_frame, rows["tune"], tune_feature, target, head
        )
        heads[target] = {
            "head": head,
            "placebo": placebo,
            "cutoff": cutoff,
            "interval_scale": interval_scale,
        }
        report["targets"][target] = {
            "cutoff": cutoff,
            "normalized_conformal_interval_scale": interval_scale,
            "training_rows": int(len(rows["train"])),
            "tuning_rows": int(len(rows["tune"])),
            "training_dynamic_fraction": head["training_dynamic_fraction"],
            "evaluation": {},
        }
        for role, label in (("test", "eicu_patient"), ("hospital", "eicu_hospital")):
            role_pos = positions(rows[role], source_lookup)
            report["targets"][target]["evaluation"][label] = evaluate(
                source_frame, rows[role], source_features[target][role_pos], target,
                head, placebo, cutoff, interval_scale, label,
            )

    del source_features
    print("prepare MIMIC-IV external features", flush=True)
    external_frame, external_rows, external_features, external_lookup = prepare_external(args, assets)
    for target in TARGETS:
        rows = external_rows[target]
        pos = positions(rows, external_lookup)
        fitted = heads[target]
        result = evaluate(
            external_frame, rows, external_features[target][pos], target,
            fitted["head"], fitted["placebo"], fitted["cutoff"], fitted["interval_scale"],
            "mimic_external",
        )
        report["targets"][target]["evaluation"]["mimic_external"] = result
        report["targets"][target]["promotion_allowed"] = bool(
            all(item["full_gate_pass"] for item in report["targets"][target]["evaluation"].values())
        )
        print(json.dumps({
            "target": target,
            "eicu_patient": report["targets"][target]["evaluation"]["eicu_patient"]["full_gate_pass"],
            "eicu_hospital": report["targets"][target]["evaluation"]["eicu_hospital"]["full_gate_pass"],
            "mimic_external": result["full_gate_pass"],
            "mimic_mae": result["mae"],
            "mimic_crps": result["crps"],
            "mimic_coverage": result["coverage"],
        }, allow_nan=False), flush=True)
    report["summary"] = {
        "evaluated_targets": len(TARGETS),
        "promoted_targets": [target for target, item in report["targets"].items() if item["promotion_allowed"]],
        "mimic_point_better_than_persistence": [
            target for target, item in report["targets"].items()
            if item["evaluation"]["mimic_external"]["mae"]["state_conditioned"]
            < item["evaluation"]["mimic_external"]["mae"]["persistence"]
        ],
        "mimic_crps_better_than_empirical_persistence": [
            target for target, item in report["targets"].items()
            if item["evaluation"]["mimic_external"]["crps"]["state_conditioned"]
            < item["evaluation"]["mimic_external"]["crps"]["empirical_persistence"]
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(
        {
            "schema": "whole_body_jepa.state_conditioned_kidney_heads.v1",
            "status": "candidate_only",
            "release_binding": report["release_binding"],
            "heads": heads,
        },
        args.output_dir / "heads.joblib",
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_findings(report, args.findings)
    print(json.dumps(report["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--eicu", type=Path,
        default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_future_label_repaired/eicu_aki_transitions_6h.parquet"),
    )
    parser.add_argument(
        "--mimic", type=Path,
        default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/mimiciv_observation_transitions_6h_full_v31_treatment_unit_safe.parquet"),
    )
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/state_conditioned_kidney_20260904_v2"))
    parser.add_argument("--findings", type=Path, default=Path("STATE_CONDITIONED_KIDNEY_FINDINGS_20260904.md"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
