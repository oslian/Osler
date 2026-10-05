"""Compare three frozen readouts on identical patients, then compile one selector."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from whole_body_jepa.multisource_intervals import sha256, stable_hash
from whole_body_jepa.source_selection import (
    BASE, HEAD, MULTI, PERSISTENCE, BOUND_FILES, patient_roles,
    predict_head_values, select_source, validate_selected,
)
from scripts.calibrate_multisource_intervals import unused_patients


def prediction_frame(frame, features, cell, target, scale, median, single, multi):
    from osler_jepa.shared_release_features import FEATURE_SLICES

    output = frame[["subject_id", "stay_id", "t", "careunit"]].copy()
    current = frame[f"{target}_t"].to_numpy(dtype=float)
    output["current"] = current
    output["truth"] = frame[f"future_{target}"].to_numpy(dtype=float)
    output[PERSISTENCE] = current
    output[BASE] = features[:, FEATURE_SLICES["base_point"].start].astype(float) * scale + median
    if cell in single["heads"]:
        output[HEAD] = predict_head_values(single["heads"][cell], features, current, runtime=True)
    output[MULTI] = predict_head_values(multi["heads"][cell], features, current, runtime=False)
    return output


def summarize(decisions: dict) -> dict:
    sources = Counter(
        "calibrated_event_risk_head" if r["target"] == "hypoxemia_event" else r["source_model"]
        for r in decisions.values()
    )
    return {"catalog_cells": len(decisions),
            "compared_cells": sum("selection" in r for r in decisions.values()),
            "changed_cells": sum(r["source_changed"] for r in decisions.values()),
            "new_audit_pass_cells": sum(r["validation"]["promotion_allowed"] for r in decisions.values()),
            "runtime_move_cells": sum(r["can_move"] for r in decisions.values()),
            "source_counts": dict(sources)}


def findings(payload: dict, path: Path) -> None:
    summary = payload["summary"]
    text = ["# Unified Frozen JEPA Source Selection", "",
            "## Result", "",
            f"- Catalog: {summary['catalog_cells']} cells; compared numeric cells: {summary['compared_cells']}.",
            f"- New source replacements: {summary['changed_cells']}.",
            f"- Selected source with both new point/interval gates: {summary['new_audit_pass_cells']}.",
            f"- Unified runtime movable cells: {summary['runtime_move_cells']} (overlaps, not additive).",
            f"- Runtime sources: `{summary['source_counts']}`.", "",
            "## Protocol", "",
            "All three readouts use one frozen checkpoint and the same 311-column representation.",
            "Base JEPA uses its value output. Original shared heads retain their exact frozen",
            "runtime normalization; multi-source heads use their fitted development normalization.",
            "There is no new weight fitting, target-domain normalization, or test-time refitting.", "",
            "For every cell, reconstruct and exclude all MIMIC patients in the multi-source",
            "head's original selection pool, not just individual fitted rows. Split remaining",
            "patients by a fixed hash into source selection (40%), calibration (30%) and",
            "final development test (30%). Source selection requires seven fixed patient",
            "split comparisons, significant improvement over the incumbent and persistence,",
            "then ranks eligible challengers by patient-equal MAE. These overlapping splits",
            "are robustness checks, not seven independent replications.", "",
            "Freeze the chosen source before reading NWICU outcomes. Calibration uses one",
            "outcome-independent anchor per calibration patient and the exact 90% order statistic.",
            "MIMIC test and NWICU must pass patient/careunit/late-time paired bootstrap versus",
            "persistence and the incumbent, plus row/patient/unit/late coverage in [0.87,0.93].",
            "Direction accuracy and delta correlation are reported as diagnostics.",
            "Failure does not trigger trying the runner-up on the external test: retain the",
            "incumbent with its own original interval and original validation scope.", "",
            "Both databases have previously been inspected during research. This is a reproducible",
            "development/transfer audit, NOT a new untouched lockbox or prospective clinical test.",
            "Careunit gates aggregate supported units; no guarantee for every small subgroup.",
            "Released timestamps are not original calendar time. No causal claim is made.", "",
            "## Per-cell decisions", "",
            "| Cell | Incumbent | Development choice | Runtime source | Changed | New joint gate | Failure |",
            "|---|---|---|---|---|---|---|"]
    for cell, row in payload["cells"].items():
        text.append(f"| {cell} | {row['incumbent_source']} | {row['selected_source']} | {row['source_model']} | "
                    f"{row['source_changed']} | {row['validation']['promotion_allowed']} | "
                    f"{', '.join(row['validation']['failure_reasons']) or '-'} |")
    text += ["", "## Serving", "",
             "`whole_body_jepa.runtime.predict_patient` loads the hash-bound selector and",
             "runs the checkpoint once per horizon. Each target reads only its selected head.",
             "The interval is inseparable from that source: new source/new calibration or",
             "retained source/original calibration. Never use the old interval for a new point.",
             "Four urine/event cells lack a matching NWICU target contract; they retain their",
             "existing models and are explicitly not counted as newly validated.",
             "Missing current values remain missing; input zero remains observed. A missing",
             "selector or mismatched weights stops loading instead of silently changing models.",
             "No next-measurement-time prediction, new clinical indication, or latent-coupling",
             "proof is introduced. All patient-level caches stay outside this repository.", ""]
    if payload.get("interval_rescue"):
        text += ["## Interval-only rescue", "",
                 "The four fallback cells with both point gates already passed were tested",
                 "with one fixed conditional scale model. It uses current value and absolute",
                 "predicted change, fits on selection patients, and calibrates on separate",
                 "patients at nominal 90%. No external fitting or point-weight changes.",
                 f"Promoted cells: `{payload['interval_rescue']['promoted_cells']}`.",
                 "Other cells retain their prior source and interval. This follow-up reuses",
                 "research validation data and is not independent confirmatory evidence.",
                 "Details: [interval rescue](PERSISTENCE_INTERVAL_RESCUE_FINDINGS_20260902.md).", ""]
    path.write_text("\n".join(text))


def run(args):
    sys.path.insert(0, str(args.research_root))
    from unified_shared_jepa_head_audit import extract_shared_features
    from multisource_shared_jepa_forecast_audit import _canonical_external_cohort
    from eicu_to_mimic_frozen_jepa_audit import load_frozen_model
    from mimic_to_eicu_49_cell_external_audit import cap_by_subject, normalize_cohort

    model_dir = args.model_dir.resolve()
    binding = {name: sha256(model_dir / name) for name in BOUND_FILES}
    model, scaler, checkpoint = load_frozen_model(model_dir / "whole_body_joint_jepa.pt")
    single = joblib.load(model_dir / "unified_shared_jepa_head_bundle.joblib")
    multi = joblib.load(model_dir / "multisource_shared_jepa_head_bundle.joblib")
    for bundle in (single, multi):
        if bundle["checkpoint_sha256"] != binding["whole_body_joint_jepa.pt"]:
            raise ValueError("readout checkpoint mismatch")
    original = json.loads((model_dir / "runtime_registry.json").read_text())
    previous = json.loads(args.previous_audit.read_text())
    previous = {row["cell"]: row for row in previous["cells"]}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.resolve().is_relative_to(model_dir.parent):
        raise ValueError("patient-level predictions must stay outside repository")
    key_common = {"binding": binding, "external_sha256": sha256(args.external_examples),
                  "script_sha256": sha256(Path(__file__)),
                  "inference_sha256": sha256(Path(__file__).parents[1] / "whole_body_jepa/source_selection.py"),
                  "max_rows": args.max_rows, "max_external_rows": args.max_external_rows}
    decisions = {}
    for i, contract in enumerate(original["cells"], 1):
        cell, target, horizon = contract["cell"], contract["target"], int(contract["horizon_hours"])
        incumbent = contract["source_model"] if contract["can_move"] else PERSISTENCE
        row = {"cell": cell, "target": target, "horizon_hours": horizon,
               "incumbent_source": incumbent, "selected_source": incumbent,
               "source_model": incumbent, "can_move": contract["can_move"],
               "source_changed": False, "use_selector_interval": False, "interval_method": "constant", "radius": None}
        print(f"[{i}/41] identical-row source audit: {cell}", flush=True)
        if cell not in multi["heads"]:
            row["validation"] = {"promotion_allowed": False,
                                 "failure_reasons": ["no_matching_external_numeric_contract"]}
            decisions[cell] = row
            continue
        path = Path(args.mimic_template.format(horizon=horizon))
        key = {**key_common, "cell": cell, "mimic_sha256": sha256(path)}
        stem = cell.replace("@", "_")
        meta = args.cache_dir / f"{stem}.json"
        dev_path, ext_path = (args.cache_dir / f"{stem}_{db}.parquet" for db in ("mimic", "nwicu"))
        cached = json.loads(meta.read_text()) if meta.exists() else {}
        if cached.get("key") == key and dev_path.exists() and ext_path.exists():
            development, external = pd.read_parquet(dev_path), pd.read_parquet(ext_path)
        else:
            raw = pd.read_parquet(path)
            selection, info = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=15000)
            for name in ("original_rows", "sampled_rows", "subjects"):
                if info[name] != previous[cell]["mimic_support"][name]:
                    raise ValueError(f"cannot reproduce frozen head training exclusions: {cell}")
            full, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
            pool = cap_by_subject(unused_patients(full, selection), args.max_rows, stable_hash("source_comparison_pool", cell))
            del raw, selection, full
            index = checkpoint["variables"].index(target)
            def predict(frame):
                if frame.empty:
                    return pd.DataFrame(columns=["subject_id", "stay_id", "t", "careunit", "truth", "current", BASE, HEAD, MULTI, PERSISTENCE])
                prepared, feature = extract_shared_features(frame, model, scaler, checkpoint, target=target,
                                                            horizon=horizon, batch_size=args.batch_size)
                return prediction_frame(prepared, feature, cell, target, float(scaler.scales[index]),
                                        float(scaler.medians[index]), single, multi)
            development = predict(pool)
            del pool
            ext, _ = _canonical_external_cohort(args.external_examples, target=target, horizon=horizon, max_rows=args.max_external_rows)
            external = predict(ext)
            del ext
            development.to_parquet(dev_path, index=False)
            external.to_parquet(ext_path, index=False)
            meta.write_text(json.dumps({"key": key}))
        selected = select_source(development.loc[patient_roles(development)["selection"]], incumbent, cell)
        validation = validate_selected(development, external, selected, cell)
        row.update({"selected_source": selected["selected_source"], "selection": selected, "validation": validation})
        if validation["promotion_allowed"]:
            row.update({"source_model": selected["selected_source"],
                        "source_changed": selected["selected_source"] != incumbent,
                        "can_move": selected["selected_source"] != PERSISTENCE,
                        "use_selector_interval": True, "radius": validation["radius"]})
        decisions[cell] = row
        print(json.dumps({"cell": cell, "choice": row["selected_source"], "runtime": row["source_model"],
                          "changed": row["source_changed"], "passed": validation["promotion_allowed"],
                          "failed": validation["failure_reasons"]}), flush=True)
    if binding != {name: sha256(model_dir / name) for name in BOUND_FILES}:
        raise ValueError("model artifacts changed during comparison")
    summary = summarize(decisions)
    payload = {"schema": "whole_body_jepa.unified_source_selector.v1", "release_binding": binding,
               "selection_database": "MIMIC-IV_unused_fitting_patients", "external_database": "NWICU",
               "new_untouched_lockbox_claim": False, "external_labels_used_for_source_selection": False,
               "calibration_level": .9, "coverage_bounds": [.87, .93],
               "causal_claim_allowed": False, "clinical_promotion_allowed": False,
               "next_measurement_time_modeled": False, "summary": summary, "cells": decisions}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    findings(payload, args.findings)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--previous-audit", type=Path, default=Path("artifacts/multisource_shared_jepa_forecast/multisource_shared_jepa_forecast_audit_20260902.json"))
    parser.add_argument("--mimic-template", default="/private/tmp/mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe.parquet")
    parser.add_argument("--external-examples", type=Path, default=Path("/private/tmp/nwicu_event_examples_41_grid.parquet"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/unified_source_predictions_20260902"))
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--max-external-rows", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("models/unified_source_selector.json"))
    parser.add_argument("--findings", type=Path, default=Path("UNIFIED_SOURCE_SELECTION_FINDINGS_20260902.md"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
