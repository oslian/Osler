"""Replay frozen release points/intervals with retrospective change diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.audit_release_masked_cells import read_examples, input_frame, select_rows
from whole_body_jepa.change_evaluation import CHANGE_THRESHOLDS, evaluate_change
from whole_body_jepa.contracts import VARIABLE_BOUNDS
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import _build_inputs, _predict_interval_scale, load_release
from whole_body_jepa.source_selection import BASE, HEAD, MULTI, PERSISTENCE, predict_head_result


def label_mask(frame, target):
    """No silent downgrade to stale/imputed/nominal-only future supervision."""
    required = [f"future_{target}", f"future_label_quality_{target}",
                f"future_delta_t_hr_{target}", f"future_event_time_{target}"]
    if not set(required).issubset(frame):
        return np.zeros(len(frame), dtype=bool)
    truth = pd.to_numeric(frame[f"future_{target}"], errors="coerce")
    dt = pd.to_numeric(frame[f"future_delta_t_hr_{target}"], errors="coerce")
    valid = (np.isfinite(frame[f"{target}_t"]) & np.isfinite(truth)
            & frame[f"future_label_quality_{target}"].isin(["exact", "strict"])
            & np.isfinite(dt) & (dt > 0)
            & (pd.to_datetime(frame[f"future_event_time_{target}"], errors="coerce") > frame.t))
    weight = f"future_label_weight_{target}"
    if weight in frame:
        valid &= pd.to_numeric(frame[weight], errors="coerce").fillna(0) > 0
    return valid.to_numpy()


def frozen_outputs(assets, arrays, frame, rows, cell):
    """Batch replay of serving readouts, including abstention and bound intervals."""
    decision = assets.source_selector["cells"][cell]
    target, source = decision["target"], decision["source_model"]
    idx = assets.checkpoint["variables"].index(target)
    organ = int(assets.model.variable_module_ids[idx])
    pieces = []
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        with torch.inference_mode():
            output = assets.model(
                torch.from_numpy(arrays["input"][arrays["history"][batch]]).float(),
                torch.from_numpy(arrays["current"][batch]).float(),
                torch.from_numpy(arrays["mask"][batch]).float(),
                torch.from_numpy(arrays["ages"][batch]).float(),
                torch.from_numpy(arrays["presence"][batch]).float(),
                torch.from_numpy(arrays["horizon"][batch]).float())
        out = {k: v.cpu().numpy() for k, v in output.items() if isinstance(v, torch.Tensor)}
        feature = np.concatenate([
            out["context_latent"], out["belief_state"], out["predicted_target_latent"][:, idx],
            out["organ_latents"][:, organ], out["value"][:, idx, None], arrays["current"][batch],
            arrays["current"][batch, idx, None]], axis=1)
        current = frame.iloc[batch][f"{target}_t"].to_numpy(float)
        point = out["value"][:, idx].astype(float) * assets.scaler.scales[idx] + assets.scaler.medians[idx]
        abstained = np.zeros(len(batch), dtype=bool)
        if source in (HEAD, MULTI):
            bundle = assets.heads if source == HEAD else assets.multisource_heads
            present = np.isfinite(current)
            if present.any():
                point[present], abstained[present] = predict_head_result(
                    bundle["heads"][cell], feature[present], current[present], runtime=source == HEAD)
        elif source == PERSISTENCE:
            point = current.copy()
        elif source != BASE:
            raise ValueError(f"unsupported continuous source: {source}")
        fresh = np.isfinite(current) & frame.iloc[batch][f"{target}_age_hr"].between(0, assets.max_age_hours).to_numpy()
        movable = decision["can_move"] & fresh & ~abstained
        point = np.where(fresh, np.where(movable, point, current), np.nan)
        lower, upper = np.full(len(batch), np.nan), np.full(len(batch), np.nan)
        interval = assets.intervals["cells"].get(cell)
        for j in np.flatnonzero(movable):
            if decision["use_selector_interval"]:
                if decision["interval_method"] == "normalized_current_predicted_delta":
                    radius = decision["normalized_radius"] * _predict_interval_scale(
                        decision["scale_model"], np.array([[current[j], abs(point[j] - current[j])]]))
                else:
                    radius = decision["radius"]
                lower[j], upper[j] = point[j] - radius, point[j] + radius
            elif not decision["source_changed"] and interval and interval.get("runtime_eligible"):
                radius = interval["normalized_radius"] * _predict_interval_scale(
                    interval["residual_scale_model"], feature[j:j + 1]) if interval.get(
                        "runtime_interval_method") == "normalized_conformal" else interval["radius"]
                lower[j], upper[j] = point[j] - radius, point[j] + radius
                lo, hi = VARIABLE_BOUNDS.get(target, (None, None))
                if lo is not None:
                    lower[j] = max(lo, lower[j])
                if hi is not None:
                    upper[j] = min(hi, upper[j])
        pieces.append(pd.DataFrame({"point": point, "lower": lower, "upper": upper}))
    return pd.concat(pieces, ignore_index=True)


def write_report(report, directory):
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Stable, Change and Uncertainty Replay", "",
             "Frozen release diagnostic, not clinical validation or promotion. No fitting or threshold selection.",
             "eICU and NWICU have been used in prior development; these are NOT untouched lockboxes.",
             "Sampling is one eligible anchor per patient per cell, selected by hash, not by change size.",
             "Actual future change defines retrospective strata only. The model never receives these strata.",
             "Stable means smaller than a research effect size, NOT healthy or safe. Thresholds are not alarms.",
             "Intervals are exactly source-bound release intervals, never refitted for stable/change subgroups.",
             "CIs use equal-patient bootstrap. A subgroup needs 30 patients for a CI. No multiple-testing promotion.",
             "Domain slices are descriptive replays, not new hospital-heldout training experiments.",
             "Missing intervals/points are explicitly counted; coverage is conditional on issued intervals.",
             "The central interval score penalizes width and misses; it is not CRPS or full-distribution validation.",
             "Binary hypoxemia is not replaced with single-time O2sat: its interval-event labels need a separate audit.", "",
             "| Database | Cell | Patients stable / changed | Stable false move | Changed signed recall | Changed MAE delta | Coverage stable / changed |",
             "|---|---|---:|---:|---:|---:|---:|"]
    def number(metric):
        value = metric.get("mean")
        return "NA" if value is None else f"{value:.3f}"
    for db, cells in report["datasets"].items():
        for cell, item in cells.items():
            if item.get("status") != "evaluated":
                continue
            score = item["primary"]
            s, d = score["stable"], score["changed"]
            lines.append(f"| {db} | {cell} | {s['patients']} / {d['patients']} | "
                         f"{number(s['false_move_rate_on_stable'])} | {number(d['signed_change_recall_including_abstention'])} | "
                         f"{number(d['mae_delta_vs_persistence'])} | {number(s['coverage_when_interval_available'])} / "
                         f"{number(d['coverage_when_interval_available'])} |")
    lines += ["", "MAE delta is model minus persistence (negative is better), in physical units.",
              "Recall requires a threshold-sized prediction in the correct direction; abstentions count as misses.",
              "Unsupported cells and all half/double-threshold sensitivities are recorded in report.json.",
              "The replay is conditional on a measured future label; it does not establish performance when no test occurs.",
              "No treatment-intent or future-label information is added to inference. No next-measurement-time model.", ""]
    (directory / "FINDINGS.md").write_text("\n".join(lines))


def run(args):
    torch.set_num_threads(4)
    assets = load_release(args.model_dir)
    bindings = {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "whole_body_jepa.change_behavior.v1", "release_binding": bindings,
              "protocol_sha256": sha256(Path(__file__).parents[1] / "CHANGE_BEHAVIOR_PROTOCOL.md"),
              "code_sha256": {p: sha256(Path(p)) for p in
                  ("scripts/audit_change_behavior.py", "whole_body_jepa/change_evaluation.py")},
              "thresholds": CHANGE_THRESHOLDS, "max_patients_per_cell": args.max_patients,
              "model_fitted": False, "runtime_modified": False, "promotion_allowed": False,
              "clinical_claim_allowed": False, "causal_claim_allowed": False,
              "untouched_lockbox": False, "complete": False, "datasets": {}, "inputs": {}}
    cells = assets.source_selector["cells"]
    for db, path, horizons, grid in (
        ("eicu", args.eicu_grid, (1, 3, 6, 12, 24, 48), True),
        ("nwicu", args.nwicu_3h, (3,), False),
    ):
        report["inputs"][db] = {"file": str(path), "sha256": sha256(path)}
        report["datasets"][db] = {c: {"status": "no_matching_horizon_input"} for c in cells}
        for horizon in horizons:
            print(f"read {db} {horizon}h", flush=True)
            frame = read_examples(path, assets, db, horizon, grid=grid)
            arrays = _build_inputs(input_frame(frame, assets.checkpoint["variables"], assets.max_age_hours), assets, horizon)
            for cell, contract in cells.items():
                if int(contract["horizon_hours"]) != horizon:
                    continue
                target = contract["target"]
                if target == "hypoxemia_event":
                    report["datasets"][db][cell] = {"status": "requires_interval_event_labels"}
                    continue
                rows = select_rows(frame, label_mask(frame, target), args.max_patients, f"change:{db}:{cell}")
                if not len(rows):
                    report["datasets"][db][cell] = {"status": "no_strict_measured_pairs"}
                    continue
                selected = frame.iloc[rows].reset_index(drop=True)
                predictions = frozen_outputs(assets, arrays, frame, rows, cell)
                scored = pd.DataFrame({"subject_id": selected.subject_id,
                    "current": selected[f"{target}_t"], "truth": pd.to_numeric(selected[f"future_{target}"], errors="coerce"),
                    "point": predictions.point, "lower": predictions.lower, "upper": predictions.upper})
                primary = evaluate_change(scored, CHANGE_THRESHOLDS[target], repetitions=args.bootstrap)
                units = {}
                for unit, indexes in selected.groupby("careunit").indices.items():
                    if len(indexes) >= 30 and unit.lower() not in ("unknown", "nan", "none", ""):
                        units[str(unit)] = evaluate_change(scored.iloc[indexes], CHANGE_THRESHOLDS[target], repetitions=args.bootstrap)
                item = {"status": "evaluated", "source": contract["source_model"],
                        "runtime_can_move": contract["can_move"], "primary": primary, "careunits": units,
                        "label_policy": "strict_or_exact_measured_positive_actual_delta_t",
                        "actual_delta_t_median_hours": float(selected[f"future_delta_t_hr_{target}"].median()),
                        "sensitivity": {str(m): evaluate_change(scored, m * CHANGE_THRESHOLDS[target], repetitions=args.bootstrap)
                                        for m in (.5, 2.)}}
                report["datasets"][db][cell] = item
                print(json.dumps({"database": db, "cell": cell, "patients": len(rows),
                    "stable": primary["stable"]["patients"], "changed": primary["changed"]["patients"]}), flush=True)
            write_report(report, args.output)
            del frame, arrays
    if bindings != {p.name: sha256(p) for p in args.model_dir.iterdir() if p.is_file()}:
        raise ValueError("release changed during audit")
    report["complete"] = True
    write_report(report, args.output)
    print("FINISHED: frozen diagnostic only; no promotions", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--eicu-grid", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_canonical_event_examples_full_5000.parquet"))
    p.add_argument("--nwicu-3h", type=Path, default=Path("/private/tmp/nwicu_event_examples_3h_full.parquet"))
    p.add_argument("--max-patients", type=int, default=500)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--output", type=Path, default=Path("artifacts/change_behavior_20260906"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
