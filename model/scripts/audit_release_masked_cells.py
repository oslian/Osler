"""Frozen release leave-variable-group-out audit, not a promotion procedure.

Same-anchor reconstruction and missing-target future prediction are separate
tasks. Never pass a hidden anchor into a residual head or reuse its intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from osler_jepa.shared_release_features import build_target_head_features
from whole_body_jepa.multisource_intervals import (
    SEEDS, _bootstrap, patient_split, representative_rows, sha256, stable_hash,
)
from whole_body_jepa.nowcast import related_siblings
from whole_body_jepa.runtime import _build_inputs, load_release
from whole_body_jepa.source_selection import BASE, HEAD, MULTI, PERSISTENCE, predict_head_values


DONOR_SEEDS = (137, 271, 409)


def read_examples(path, assets, database, horizon, *, grid=False):
    names = pq.ParquetFile(path).schema_arrow.names
    variables = assets.checkpoint["variables"]
    metadata = {"subject_id", "stay_id", "t", "anchor_time", "hours_since_onset",
                "careunit", "first_careunit", "unittype", "horizon_hours", "hospitalid"}
    wanted = metadata | {f"{v}{s}" for v in variables for s in ("_t", "_age_hr", f"_tp{horizon}")}
    wanted |= {f"{p}{v}" for v in variables for p in
               ("future_", "future_label_quality_", "future_label_weight_",
                "future_event_time_", "future_delta_t_hr_")}
    frame = pd.read_parquet(path, columns=[n for n in names if n in wanted],
                            filters=[("horizon_hours", "=", float(horizon))] if grid else None)
    frame["subject_id"] = database + ":" + frame["subject_id"].astype(str)
    frame["stay_id"] = database + ":" + frame["stay_id"].astype(str)
    frame["t"] = pd.to_datetime(frame["t"] if "t" in frame else frame["anchor_time"], errors="coerce")
    frame = frame.loc[frame["t"].notna()].copy()
    unit = next((c for c in ("first_careunit", "careunit", "unittype") if c in frame), None)
    frame["careunit"] = frame[unit].fillna("unknown").astype(str) if unit else "unknown"
    frame = frame.sort_values(["stay_id", "t"], kind="stable").drop_duplicates(["stay_id", "t"]).reset_index(drop=True)
    origin = frame.groupby("stay_id")["t"].transform("min")
    frame["_time_hours"] = (frame["t"] - origin).dt.total_seconds() / 3600
    if "hours_since_onset" not in frame:
        frame["hours_since_onset"] = frame["_time_hours"]
    else:
        frame["hours_since_onset"] = pd.to_numeric(frame["hours_since_onset"], errors="coerce").fillna(0)
    columns = {c: frame[c] for c in frame}
    for v in variables:
        for suffix in ("_t", "_age_hr"):
            col = f"{v}{suffix}"
            columns[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame else np.nan
        if f"{v}_tp{horizon}" in frame:
            columns[f"future_{v}"] = pd.to_numeric(frame[f"{v}_tp{horizon}"], errors="coerce")
    return pd.DataFrame(columns, index=frame.index)


def input_frame(frame, variables, max_age):
    """An explicit whitelist prevents labels, IDs and workflow entering encoding."""
    columns = {c: frame[c] for c in ("subject_id", "stay_id", "hours_since_onset", "_time_hours")}
    for v in variables:
        age = frame[f"{v}_age_hr"]
        columns[f"{v}_t"] = frame[f"{v}_t"].where(age.between(0, max_age))
        columns[f"{v}_age_hr"] = age
    return pd.DataFrame(columns, index=frame.index)


def mask_inputs(arrays, hidden):
    # Serving neutralizes measurement-process masks/ages. Retain that contract;
    # zero here is the checkpoint's normalized missing-value prior, not raw zero.
    view = dict(arrays)
    view["input"] = arrays["input"].copy()
    view["current"] = arrays["current"].copy()
    view["input"][:, hidden] = 0.0
    view["current"][:, hidden] = 0.0
    return view


def select_rows(frame, valid, limit, salt):
    eligible = np.flatnonzero(np.asarray(valid))
    if not len(eligible):
        return eligible
    local = representative_rows(frame.iloc[eligible], stable_hash("masked_anchor", salt))
    rows = eligible[local]
    hashes = [stable_hash("masked_patient", salt, p) for p in frame.iloc[rows]["subject_id"]]
    return rows[np.argsort(hashes, kind="stable")[:limit]]


def donor_indices(subjects, seed):
    """A derangement of one-anchor-per-patient contexts, never their outcomes."""
    subjects = np.asarray(subjects).astype(str)
    if len(np.unique(subjects)) != len(subjects) or len(subjects) < 2:
        raise ValueError("donor control requires distinct patients")
    order = np.random.default_rng(seed).permutation(len(subjects))
    donors = np.empty(len(subjects), dtype=int)
    donors[order] = np.roll(order, 1)
    assert np.all(subjects != subjects[donors])
    return donors


def valid_future_rows(frame, target):
    truth = pd.to_numeric(frame[f"future_{target}"], errors="coerce")
    valid = np.isfinite(truth) & np.isfinite(frame[f"{target}_t"])
    quality = f"future_label_quality_{target}"
    if quality in frame:
        # Native ledger calls within-target-tolerance measured pairs "strict";
        # the older transition extractor calls its accepted pairs "exact".
        valid &= frame[quality].astype(str).str.lower().isin({"exact", "strict"})
    weight = f"future_label_weight_{target}"
    if weight in frame:
        valid &= pd.to_numeric(frame[weight], errors="coerce").fillna(0.0) > 0.0
    future_time = f"future_event_time_{target}"
    if future_time in frame:
        valid &= pd.to_datetime(frame[future_time], errors="coerce") > frame["t"]
    return valid


def score(frame, truth, prediction, prior, salt):
    if len(frame) < 200:
        return {"status": "insufficient_support", "patients": len(frame), "diagnostic_pass": False}
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        return {"status": "invalid_prediction", "diagnostic_pass": False}
    error = np.abs(prediction - truth)
    references = {"population": np.full(len(frame), prior)}
    for seed in DONOR_SEEDS:
        references[f"wrong_patient_{seed}"] = prediction[donor_indices(frame["subject_id"], seed)]
    units = frame["careunit"].to_numpy()
    unit_names = [u for u in sorted(set(units)) if u.lower() not in {"unknown", "nan", "none", ""}
                  and np.sum(units == u) >= 30]
    comparisons = {}
    for name, reference in references.items():
        delta = error - np.abs(reference - truth)
        splits = []
        for seed in SEEDS:
            _, test = patient_split(frame["subject_id"], seed)
            result = _bootstrap(delta[test], stable_hash(salt, name, seed)) if test.sum() >= 100 else {"pass": False}
            splits.append({"seed": seed, **result})
        unit_scores = {u: _bootstrap(delta[units == u], stable_hash(salt, name, u)) for u in unit_names}
        care_pass = len(unit_scores) >= 2 and all(s["pass"] for s in unit_scores.values())
        comparisons[name] = {"mae": float(np.mean(np.abs(reference - truth))),
                             "paired": _bootstrap(delta, stable_hash(salt, name, "all")),
                             "patient_splits": splits, "careunits": unit_scores,
                             "pass": all(s["pass"] for s in splits) and care_pass}
    return {"status": "evaluated", "patients": len(frame), "mae": float(error.mean()),
            "comparisons": comparisons, "diagnostic_pass": all(c["pass"] for c in comparisons.values())}


def infer(assets, arrays, rows, target, cell=None, batch_size=256):
    idx = assets.checkpoint["variables"].index(target)
    organ = int(assets.model.variable_module_ids[idx])
    scale, median = float(assets.scaler.scales[idx]), float(assets.scaler.medians[idx])
    predictions = {"reconstruction": []} if cell is None else {BASE: []}
    if cell in assets.heads["heads"]:
        predictions[HEAD] = []
    if cell in assets.multisource_heads["heads"]:
        predictions[MULTI] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            current = torch.from_numpy(arrays["current"][batch]).float()
            args = (torch.from_numpy(arrays["input"][arrays["history"][batch]]).float(), current,
                    torch.from_numpy(arrays["mask"][batch]).float(),
                    torch.from_numpy(arrays["ages"][batch]).float(),
                    torch.from_numpy(arrays["presence"][batch]).float())
            if cell is None:
                latent = assets.model.encode_context_components(*args)[0]
                rec = assets.model.reconstruction_heads[idx](latent).cpu().numpy().ravel()
                predictions["reconstruction"].append(rec * scale + median)
                continue
            out = assets.model(*args, torch.from_numpy(arrays["horizon"][batch]).float())
            base = out["value"][:, idx].cpu().numpy()
            predictions[BASE].append(base * scale + median)
            if len(predictions) == 1:
                continue
            arrays_out = {k: out[k].cpu().numpy() for k in
                          ("context_latent", "belief_state", "predicted_target_latent", "organ_latents")}
            feature = np.stack([build_target_head_features(
                shared_context=arrays_out["context_latent"][i], patient_belief=arrays_out["belief_state"][i],
                target_future=arrays_out["predicted_target_latent"][i, idx],
                target_organ=arrays_out["organ_latents"][i, organ], base_forecast_normalized=base[i],
                measurement_pure_state=arrays["current"][row],
                target_current_normalized=arrays["current"][row, idx]) for i, row in enumerate(batch)])
            # This anchor comes from masked arrays, never from held-out truth.
            anchor = arrays["current"][batch, idx].astype(float) * scale + median
            for source, bundle, runtime in ((HEAD, assets.heads, True), (MULTI, assets.multisource_heads, False)):
                if source in predictions:
                    predictions[source].append(predict_head_values(bundle["heads"][cell], feature, anchor, runtime=runtime))
    return {source: np.concatenate(values).astype(float) for source, values in predictions.items()}


def describe_inputs(frame, rows, variables, target):
    hidden = sorted(({target} | related_siblings(target)) & set(variables))
    return {"hidden_variables": hidden, "history_rows": len(frame), "evaluated_patients": len(rows),
            "fresh_truth_max_age_hours": 1.0,
            "current_input_variables_available": int(sum(frame[f"{v}_t"].notna().any() for v in variables))}


def write_report(payload, output, findings):
    summary = {}
    for db in ("mimic", "nwicu"):
        current = {t: r[db] for t, r in payload["reconstruction"].items() if db in r}
        future = {c: r["future"][db] for c, r in payload["cells"].items() if db in r["future"]}
        summary[db] = {
            "current_targets_evaluated": sum(r.get("status") == "evaluated" for r in current.values()),
            "current_targets_passing": [t for t, r in current.items() if r.get("diagnostic_pass")],
            "current_targets_all_seven_patient_splits": [t for t, r in current.items()
                if r.get("status") == "evaluated" and all(
                    all(s["pass"] for s in v["patient_splits"]) for v in r["comparisons"].values())],
            "future_cells_evaluated": sum(r.get("status") == "evaluated" for r in future.values()),
            "future_cell_source_passes": [[c, s] for c, r in future.items()
                for s, v in r.get("sources", {}).items() if v["diagnostic_pass"]],
        }
    payload["summary"] = summary
    output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    rec = payload["reconstruction"]
    lines = ["# Frozen Release Masked-Cell Audit", "", "## Result", "",
             f"Catalog: {len(payload['cells'])} cells, corresponding to 15 current numeric variables plus an interval-event target.",
             f"MIMIC: {summary['mimic']['current_targets_evaluated']} current targets and {summary['mimic']['future_cells_evaluated']} numeric forecast cells evaluated.",
             f"NWICU: {summary['nwicu']['current_targets_evaluated']} current targets and {summary['nwicu']['future_cells_evaluated']} numeric forecast cells evaluated.",
             f"Full current reconstruction diagnostic passes: MIMIC `{summary['mimic']['current_targets_passing']}`; NWICU `{summary['nwicu']['current_targets_passing']}`.",
             f"Seven patient-split passes before the care-unit gate: MIMIC `{summary['mimic']['current_targets_all_seven_patient_splits']}`; NWICU `{summary['nwicu']['current_targets_all_seven_patient_splits']}`.",
             f"Masked forecast source passes: MIMIC `{summary['mimic']['future_cell_source_passes']}`; NWICU `{summary['nwicu']['future_cell_source_passes']}`.",
             "The two event cells require a separate event label contract. Urine has no NWICU label;",
             "platelets@1h has no eligible unused MIMIC patients in the frozen head exclusion pool.",
             "Unsupported cells are not model failures. No candidate is promoted by this audit.", "",
             "## Protocol", "",
             "All weights, source choices and scalers are frozen; no runtime changes or promotions.",
             "Mask the target and predefined siblings from every input/history row. Recompute",
             "context, belief, base point, organ and target features. Missing anchors use the",
             "checkpoint population prior, not the hidden value. Serving's neutral workflow",
             "mask/age contract is preserved. Labels, provenance and future treatments are not inputs.",
             f"One hash-selected eligible anchor per patient, at most {payload['max_patients_per_task']} patients per task; not a full-row census.",
             "Same-anchor truth must be measured within 1 hour. Future labels are separate.",
             "Compare with the frozen population prior and three wrong-patient context controls.",
             "The latter are same-frozen-model derangements, NOT separately trained shuffled-label models.",
             "Require seven overlapping patient split CIs and every supported careunit CI below zero.",
             "Units need 30 patients; at least two units. Splits need 100 test patients.",
             "These are nominal diagnostic screens, not multiplicity-corrected promotion gates.",
             "Existing forecast intervals are invalid under masking and are not reused.",
             "Both databases have been inspected previously; neither is an untouched lockbox.", "",
             "## Same-anchor reconstruction", "",
             "This tests the packaged checkpoint's reconstruction head, NOT old retrained candidates.",
             "A target's current reconstruction result is reused across horizons, not counted repeatedly.", "",
             "| Target | MIMIC MAE / population MAE | MIMIC diagnostic | NWICU MAE / population MAE | NWICU diagnostic |",
             "|---|---|---|---|---|"]
    def val(s):
        return f"{s['mae']:.4g} / {s['comparisons']['population']['mae']:.4g}" if s.get("status") == "evaluated" else s.get("status", "missing")
    for target, dbs in sorted(rec.items()):
        a, b = dbs.get("mimic", {}), dbs.get("nwicu", {})
        lines.append(f"| {target} | {val(a)} | {a.get('diagnostic_pass', False)} | {val(b)} | {b.get('diagnostic_pass', False)} |")
    lines += ["", "MAE is in the target's physical units, not a percentage accuracy.", "",
              "## All catalog cells", "",
              "Forecast heads do not define a same-anchor nowcast. Their forced missing-anchor",
              "predictions below are stress tests, not new serving capabilities. Persistence",
              "cannot be computed without its anchor; unmasked persistence is diagnostic only.", "",
              "| Cell | Runtime source | Current reconstruction (MIMIC / NWICU) | Masked future sources passing MIMIC | Passing NWICU |",
              "|---|---|---|---|---|"]
    for cell, row in payload["cells"].items():
        target = row["target"]
        current = rec.get("o2sat" if target == "hypoxemia_event" else target, {})
        outcomes = []
        for db in ("mimic", "nwicu"):
            result = row.get("future", {}).get(db, {})
            if result.get("status") != "evaluated":
                outcomes.append(result.get("status", "not_run"))
            else:
                outcomes.append(", ".join(k for k, v in result["sources"].items() if v["diagnostic_pass"]) or "none")
        curr = " / ".join(str(current.get(db, {}).get("diagnostic_pass", False)) for db in ("mimic", "nwicu"))
        lines.append(f"| {cell} | {row['source_model']} | {curr} | {outcomes[0]} | {outcomes[1]} |")
    lines += ["", "## Interpretation limits", "",
              "Failure can reflect missing patient-specific information or masked-input distribution",
              "shift; it does not prove a variable is fundamentally unpredictable. Success supports",
              "conditional recoverability, not complete physiology, causality or hidden-state truth.",
              "The shipped model was not block-mask trained identically to the newer reconstruction",
              "candidates. Their historic passes cannot be attributed to this checkpoint.",
              "Urine lacks a matching NWICU output contract. Binary hypoxemia requires its own",
              "interval-event labels; current O2sat reconstruction is not event-risk validation.", ""]
    lines += ["## What failed", "",
              "The frozen checkpoint was trained for forecasts anchored to an observed target.",
              "Its reconstruction branch did not receive the newer, matched full-history/sibling",
              "block-mask pretraining. This audit tests that packaged branch, not those later",
              "reconstruction candidates. Contract mismatch is a plausible contributor, not a",
              "proven sole cause. Low population MAE differences alone are not reliable recovery.",
              "Hematocrit and platelets show patient-level recovery evidence in both databases,",
              "but not significant improvements in every supported care unit. A failed unit CI",
              "may cross zero; it does not necessarily mean the point estimate regressed.",
              "The all-supported-unit diagnostic gate here is stricter than the pooled/equal-unit",
              "forecast serving gate. Counts are therefore not directly comparable to the release's",
              "28 movable cells. Removing the entire target history is also a different information",
              "setting from normal observed-anchor forecasting and may be out of training support.",
              "Thus this result does not revoke normal forecast validation, and does not prove",
              "the latent contains no signal. It rejects universal masked-input recovery for this release.", "",
              "## Reproduce", "", "```bash", "python -m scripts.audit_release_masked_cells", "```", "",
              "Override the documented CLI paths for local canonical examples and unused-patient",
              "cache. Model and input hashes are recorded. Only aggregate results are written.", ""]
    findings.write_text("\n".join(lines))


def run(args):
    torch.set_num_threads(args.threads)
    assets = load_release(args.model_dir)
    model_dir = Path(args.model_dir)
    bindings = {p.name: sha256(p) for p in model_dir.iterdir() if p.is_file()}
    cells = {k: {"target": v["target"], "horizon_hours": v["horizon_hours"],
                 "source_model": v["source_model"], "future": {}}
             for k, v in assets.source_selector["cells"].items()}
    variables = assets.checkpoint["variables"]
    targets = sorted({"o2sat" if r["target"] == "hypoxemia_event" else r["target"] for r in cells.values()})
    payload = {"schema": "whole_body_jepa.frozen_release_masked_audit.v1", "release_binding": bindings,
               "audit_script_sha256": sha256(Path(__file__)),
               "causal_claim_allowed": False, "runtime_modified": False, "promotion_allowed": False,
               "next_measurement_time_modeled": False, "new_untouched_lockbox_claim": False,
               "max_patients_per_task": args.max_patients, "reconstruction": {}, "cells": cells,
               "input_files": {}}
    for database, path in (("mimic", args.mimic_current), ("nwicu", args.nwicu_current)):
        payload["input_files"][str(path)] = sha256(path)
        frame = read_examples(path, assets, database, 3)
        arrays = _build_inputs(input_frame(frame, variables, assets.max_age_hours), assets, 3)
        for target in targets:
            valid = frame[f"{target}_t"].notna() & frame[f"{target}_age_hr"].between(0, 1)
            rows = select_rows(frame, valid, args.max_patients, target)
            info = describe_inputs(frame, rows, variables, target)
            print(f"current {database} {target}: {len(rows)} patients", flush=True)
            if len(rows) < 200:
                result = {"status": "insufficient_fresh_observations", "diagnostic_pass": False}
            else:
                idx = variables.index(target)
                hidden = [variables.index(v) for v in info["hidden_variables"]]
                pred = infer(assets, mask_inputs(arrays, hidden), rows, target, batch_size=args.batch_size)["reconstruction"]
                result = score(frame.iloc[rows], frame.iloc[rows][f"{target}_t"].to_numpy(), pred,
                               float(assets.scaler.medians[idx]), f"{database}:{target}:current")
                unmasked = infer(assets, arrays, rows, target, batch_size=args.batch_size)["reconstruction"]
                result["unmasked_reconstruction_mae_diagnostic_only"] = float(np.mean(np.abs(
                    unmasked - frame.iloc[rows][f"{target}_t"].to_numpy())))
            payload["reconstruction"].setdefault(target, {})[database] = {**info, **result}
            write_report(payload, args.output, args.findings)
        del frame, arrays

    for horizon in sorted({int(r["horizon_hours"]) for r in cells.values()}):
        for database in ("mimic", "nwicu"):
            path = Path(args.mimic_template.format(horizon=horizon)) if database == "mimic" else args.nwicu_future
            if str(path) not in payload["input_files"]:
                payload["input_files"][str(path)] = sha256(path)
            frame = read_examples(path, assets, database, horizon, grid=database == "nwicu")
            arrays = _build_inputs(input_frame(frame, variables, assets.max_age_hours), assets, horizon)
            for cell, row in cells.items():
                if int(row["horizon_hours"]) != horizon:
                    continue
                target = row["target"]
                if target == "hypoxemia_event":
                    row["future"][database] = {"status": "requires_binary_interval_event_labels"}
                    continue
                if f"future_{target}" not in frame:
                    row["future"][database] = {"status": "no_compatible_future_target"}
                    continue
                truth = pd.to_numeric(frame[f"future_{target}"], errors="coerce")
                valid = valid_future_rows(frame, target)
                # Reuse the prior audit's patient exclusions, not its predictions.
                # Shared multi-source heads were fitted on a subset of MIMIC.
                if database == "mimic" and cell in assets.multisource_heads["heads"]:
                    cache = args.previous_cache / f"{cell.replace('@', '_')}_mimic.parquet"
                    if not cache.exists():
                        row["future"][database] = {"status": "training_exclusions_unavailable"}
                        continue
                    meta = json.loads((args.previous_cache / f"{cell.replace('@', '_')}.json").read_text())
                    if meta["key"]["binding"] != assets.source_selector["release_binding"]:
                        raise ValueError("patient-exclusion cache belongs to another release")
                    if meta["key"]["mimic_sha256"] != payload["input_files"][str(path)]:
                        raise ValueError("patient-exclusion input file changed")
                    allowed = pd.read_parquet(cache, columns=["subject_id"])["subject_id"].astype(str).str.replace("MIMIC:", "mimic:", regex=False)
                    valid &= frame["subject_id"].isin(set(allowed))
                rows = select_rows(frame, valid, args.max_patients, cell)
                print(f"future {database} {cell}: {len(rows)} patients", flush=True)
                if len(rows) < 200:
                    row["future"][database] = {"status": "insufficient_supported_patients", "patients": len(rows)}
                    continue
                info = describe_inputs(frame, rows, variables, target)
                hidden = [variables.index(v) for v in info["hidden_variables"]]
                masked = mask_inputs(arrays, hidden)
                predictions = infer(assets, masked, rows, target, cell, args.batch_size)
                idx = variables.index(target)
                selected_frame = frame.iloc[rows]
                y = truth.iloc[rows].to_numpy()
                sources = {s: score(selected_frame, y, p, float(assets.scaler.medians[idx]),
                                    f"{database}:{cell}:{s}") for s, p in predictions.items()}
                row["future"][database] = {"status": "evaluated", **info, "sources": sources,
                    "unmasked_persistence_mae_diagnostic_only": float(np.mean(np.abs(y - selected_frame[f"{target}_t"].to_numpy()))),
                    "label_policy": "exact_or_strict_when_available_otherwise_existing_transition_contract",
                    "interval_status": "not_calibrated_for_masked_inputs"}
                write_report(payload, args.output, args.findings)
            del frame, arrays
    if bindings != {p.name: sha256(p) for p in model_dir.iterdir() if p.is_file()}:
        raise ValueError("release changed during frozen audit")
    payload["complete"] = True
    write_report(payload, args.output, args.findings)
    print(json.dumps({"complete": True, "catalog_cells": len(cells), "unique_current_targets": len(targets),
                      "reconstruction_pass_both": [t for t, dbs in payload["reconstruction"].items()
                                                   if all(dbs[db]["diagnostic_pass"] for db in ("mimic", "nwicu"))]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--mimic-current", type=Path, default=Path("/private/tmp/mimiciv_observation_transitions_3h_full_v31_treatment_unit_safe.parquet"))
    parser.add_argument("--nwicu-current", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet"))
    parser.add_argument("--mimic-template", default="/private/tmp/mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe.parquet")
    parser.add_argument("--nwicu-future", type=Path, default=Path("/private/tmp/nwicu_event_examples_41_grid.parquet"))
    parser.add_argument("--previous-cache", type=Path, default=Path("/private/tmp/unified_source_predictions_20260902"))
    parser.add_argument("--max-patients", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("artifacts/release_masked_cells_20260903.json"))
    parser.add_argument("--findings", type=Path, default=Path("RELEASE_MASKED_CELLS_FINDINGS_20260903.md"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
