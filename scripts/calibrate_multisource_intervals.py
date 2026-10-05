"""Calibrate all frozen multi-source heads without reusing fitting patients.

Reconstruct the exact original MIMIC selection pool, exclude every patient in
that pool, and use only the remaining MIMIC patients for residual calibration.
NWICU labels are never used to choose radii. Local residual caches contain
patient identifiers and must remain outside the repository.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from whole_body_jepa.multisource_intervals import (
    BOUNDS, LEVEL, SEEDS, audit_intervals, sha256, stable_hash, subject_digest,
)


def unused_patients(full: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    reserved = set(selection["subject_id"].astype(str))
    output = full.loc[~full["subject_id"].astype(str).isin(reserved)].copy()
    if set(output["subject_id"].astype(str)) & reserved:
        raise ValueError("point-training patients leaked into interval calibration")
    return output


def _residuals(frame, features, head, target, predict_head):
    current = pd.to_numeric(frame[f"{target}_t"], errors="raise").to_numpy(dtype=float)
    truth = pd.to_numeric(frame[f"future_{target}"], errors="raise").to_numpy(dtype=float)
    point = predict_head(head, features, current)
    if not np.isfinite(point).all():
        raise ValueError("frozen head produced nonfinite predictions")
    output = frame[["subject_id", "stay_id", "t", "careunit"]].copy()
    output["current"], output["truth"], output["point"] = current, truth, point
    output["residual"] = np.abs(truth - point)
    return output


def write_findings(payload: dict, path: Path) -> None:
    def number(value):
        return "unavailable" if value is None else f"{value:.4f}"

    lines = [
        "# Multi-source 90% interval calibration", "",
        f"- Frozen point heads attempted: {payload['attempted_cells']}/41.",
        f"- Corrected NWICU point gate: {len(payload['corrected_point_pass_cells'])} cells.",
        f"- Point and interval gates together: {len(payload['runtime_eligible_cells'])} cells.",
        "- Point model weights and the existing production release were not changed.", "",
        "## Correction to the previous 37-cell claim", "",
        "The previous NWICU normalization replaced real care-unit labels with",
        "`unknown`. A single-group bootstrap was incorrectly counted as a",
        "care-unit pass. This audit restores the original labels, requires at",
        "least two supported units, and weights patients equally within units.",
        "The original 37-point-pass / 5-latent-pass claim is not a complete",
        "external gate result. Latent-vs-measurement ablations must also be rerun",
        "with corrected care-unit provenance before asserting unique latent gain.", "",
        "## Frozen calibration protocol", "",
        "1. Reproduce the original 15,000-row-cap MIMIC selection pool for each head.",
        "2. Exclude ALL patients in that pool, including their other stays and anchors.",
        "3. Use seven predeclared patient splits in the unused MIMIC pool. Choose",
        "   one outcome-independent anchor per calibration patient and the exact",
        "   ceil((n+1)*0.90) order statistic of absolute errors.",
        "4. The deployable radius uses seed 7, fixed before evaluation. No seed,",
        "   nominal level, radius, or model is selected using NWICU residuals.",
        "5. Check both row and patient-equal coverage, supported-careunit pooled",
        "   and equal coverage, and late-time coverage against [0.87, 0.93].",
        "6. A purged MIMIC early/late stress test excludes shared patients and",
        "   crossing future-label windows. The time axis is released timestamps,",
        "   not a claim of prospective real-calendar generalization.",
        "7. Test the fixed MIMIC radius on NWICU without recalibration. NWICU point",
        "   results had already been inspected; this is not a newly untouched lockbox.", "",
        "Intervals concern individual future observations, not confidence intervals",
        "for a population mean. Nominal coverage is marginal under exchangeability;",
        "repeated-window and cross-database coverage are empirical tests. Passing",
        "the aggregate care-unit gate does not guarantee 90% in every small subgroup.", "",
        "## All attempted cells", "",
        "| Cell | Radius (+/-) | Full width | NWICU row coverage | NWICU patient coverage | Corrected point gate | Point + interval | Failures |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for cell, row in payload["cells"].items():
        ext = row["nwicu_patient"]
        lines.append(f"| {cell} | {number(row['radius'])} | {number(row['mean_width'])} | "
                     f"{number(ext.get('row_coverage'))} | {number(ext.get('patient_equal_coverage'))} | "
                     f"{row['corrected_nwicu_point_gate']['pass']} | {row['runtime_interval_eligible']} | "
                     f"{', '.join(row['failure_reasons']) or '-'} |")
    lines += ["", "## Usage and boundary", "",
        "The new `models/multisource_interval_calibrator.json` binds only to the",
        "frozen checkpoint, target-head bundle, and feature contract by SHA256.",
        "The candidate registry is compiled FROM this audit; it is not a circular",
        "dependency of the calibrator. Never apply these radii to the old release.", "",
        "```python",
        "from pathlib import Path",
        "from whole_body_jepa.multisource_intervals import load_calibrator, prediction_interval",
        "calibrator = load_calibrator(Path('models/multisource_interval_calibrator.json'), Path('models'))",
        "# point must come from this bound multi-source head for this patient/horizon.",
        "interval = prediction_interval(calibrator, 'map@1h', point)",
        "```", "",
        "Failed, unsupported, or missing cells return null bounds. Patient-level",
        "residual caches stay under /private/tmp and are not release artifacts.",
        "No next-measurement-time model, causal claim, or clinical promotion was added.", "",
        "Method references: [Angelopoulos and Bates, split conformal tutorial](https://github.com/aangelopoulos/conformal-prediction)",
        "and [Dunn et al., hierarchical repeated observations](https://arxiv.org/abs/1809.07441).",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(args):
    # Use the same feature builder and normalizer as the frozen training run.
    # These research imports are not dependencies of the serving calibrator.
    sys.path.insert(0, str(args.research_root))
    from unified_shared_jepa_head_audit import extract_shared_features, predict_head
    from multisource_shared_jepa_forecast_audit import _canonical_external_cohort
    from eicu_to_mimic_frozen_jepa_audit import load_frozen_model
    from mimic_to_eicu_49_cell_external_audit import cap_by_subject, normalize_cohort

    model_dir = args.model_dir.resolve()
    names = ("whole_body_joint_jepa.pt", "multisource_shared_jepa_head_bundle.joblib", "FEATURE_CONTRACT.json")
    binding = {name: sha256(model_dir / name) for name in names}
    bundle = joblib.load(model_dir / names[1])
    if bundle.get("schema") != "multisource_shared_jepa_head_bundle.v1":
        raise ValueError("wrong point model bundle")
    if bundle["checkpoint_sha256"] != binding["whole_body_joint_jepa.pt"]:
        raise ValueError("point bundle/checkpoint mismatch")
    prior = json.loads(args.point_audit.read_text())
    prior_rows = {row["cell"]: row for row in prior["cells"]}
    model, scaler, checkpoint = load_frozen_model(model_dir / names[0])
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir.resolve().is_relative_to(model_dir.parent):
        raise ValueError("row-level residual caches must stay outside the repository")
    source_hashes = {str(Path(inspect.getfile(fn)).resolve()): sha256(Path(inspect.getfile(fn)))
                     for fn in (extract_shared_features, predict_head, normalize_cohort, _canonical_external_cohort)}
    common_key = {"models": binding, "sources": source_hashes, "batch_size": args.batch_size,
                  "max_calibration_pool_rows": args.max_rows, "max_external_rows": args.max_external_rows,
                  "point_audit_sha256": sha256(args.point_audit),
                  "external_data_sha256": sha256(args.external_examples)}
    rows = {}
    for index, (cell, head) in enumerate(bundle["heads"].items(), 1):
        print(f"[{index}/{len(bundle['heads'])}] frozen calibration: {cell}", flush=True)
        target, horizon = head["target"], int(head["horizon_hours"])
        mimic_path = Path(args.mimic_template.format(horizon=horizon))
        key = {**common_key, "cell": cell, "mimic_data_sha256": sha256(mimic_path),
               "selection_max_rows": args.selection_max_rows}
        stem = cell.replace("@", "_")
        cache_meta = args.cache_dir / f"{stem}.json"
        mimic_cache = args.cache_dir / f"{stem}_mimic.parquet"
        external_cache = args.cache_dir / f"{stem}_nwicu.parquet"
        cached = json.loads(cache_meta.read_text()) if cache_meta.exists() else {}
        if cached.get("key") == key and mimic_cache.exists() and external_cache.exists():
            mimic = pd.read_parquet(mimic_cache)
            external = pd.read_parquet(external_cache)
            support = cached["support"]
        else:
            raw = pd.read_parquet(mimic_path)
            selection, selected_info = normalize_cohort(raw, database="MIMIC-IV", target=target,
                                                       horizon=horizon, max_rows=args.selection_max_rows)
            expected = prior_rows[cell]["mimic_support"]
            for name in ("original_rows", "sampled_rows", "subjects"):
                if selected_info[name] != expected[name]:
                    raise ValueError(f"{cell}: cannot reproduce point selection pool ({name})")
            full, _ = normalize_cohort(raw, database="MIMIC-IV", target=target, horizon=horizon, max_rows=0)
            unused = unused_patients(full, selection)
            pool = cap_by_subject(unused, args.max_rows, stable_hash("interval_pool", cell))
            assert not set(pool["subject_id"]) & set(selection["subject_id"])
            support = {"excluded_point_selection_patients": int(selection["subject_id"].nunique()),
                       "excluded_subject_digest": subject_digest(selection["subject_id"]),
                       "unused_patients_available": int(unused["subject_id"].nunique()),
                       "interval_pool_patients": int(pool["subject_id"].nunique()),
                       "interval_pool_rows": len(pool),
                       "interval_pool_subject_digest": subject_digest(pool["subject_id"]),
                       "point_training_patient_overlap": 0,
                       "exclusion_scope": "all_patients_in_this_head_selection_pool_before_source_balancing"}
            del raw, full, unused, selection
            if pool.empty:
                mimic = pd.DataFrame(columns=["subject_id", "stay_id", "t", "careunit", "current", "truth", "point", "residual"])
            else:
                prepared, feature = extract_shared_features(pool, model, scaler, checkpoint, target=target,
                                                            horizon=horizon, batch_size=args.batch_size)
                mimic = _residuals(prepared, feature, head, target, predict_head)
                del prepared, feature
            del pool
            ext, ext_support = _canonical_external_cohort(args.external_examples, target=target,
                                                         horizon=horizon, max_rows=args.max_external_rows)
            prepared, feature = extract_shared_features(ext, model, scaler, checkpoint, target=target,
                                                        horizon=horizon, batch_size=args.batch_size)
            external = _residuals(prepared, feature, head, target, predict_head)
            support["external"] = {**ext_support, "careunits": int(external["careunit"].nunique())}
            del prepared, feature, ext
            mimic.to_parquet(mimic_cache, index=False)
            external.to_parquet(external_cache, index=False)
            cache_meta.write_text(json.dumps({"key": key, "support": support}, indent=2))
        result = audit_intervals(mimic, external, cell, horizon)
        rows[cell] = {"target": target, "horizon_hours": horizon, "support": support, **result}
        print(json.dumps({"cell": cell, "radius": result["radius"],
                          "nwicu_coverage": result["nwicu_patient"].get("row_coverage"),
                          "eligible": result["runtime_interval_eligible"],
                          "failed": result["failure_reasons"]}), flush=True)

    if binding != {name: sha256(model_dir / name) for name in names}:
        raise ValueError("point model changed while calibration was running")
    payload = {"schema": "whole_body_jepa.multisource_interval_calibrator.v1",
               "method": "single_outcome_independent_anchor_per_calibration_patient_split_conformal",
               "level": LEVEL, "accepted_coverage_range": list(BOUNDS),
               "primary_split_seed": SEEDS[0], "patient_split_seeds": list(SEEDS),
               "release_binding": binding, "calibration_database": "MIMIC-IV_unused_point_training_patients",
               "external_database": "NWICU", "external_labels_used_for_calibration": False,
               "external_point_results_previously_inspected": True,
               "new_untouched_lockbox_claim": False,
               "coverage_scope": "marginal_random_patient_anchor; repeated-window/domain coverage empirically audited",
               "cross_database_distribution_free_guarantee": False,
               "point_weights_changed": False, "causal_claim_allowed": False,
               "clinical_promotion_allowed": False,
               "catalog_cells": prior["catalog_cells"], "attempted_cells": len(rows),
               "runtime_eligible_cells": sorted(c for c, r in rows.items() if r["runtime_interval_eligible"]),
               "corrected_point_pass_cells": sorted(c for c, r in rows.items() if r["corrected_nwicu_point_gate"]["pass"]),
               "cells": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    write_findings(payload, args.findings)
    print(json.dumps({"attempted": len(rows), "corrected_point_pass": len(payload["corrected_point_pass_cells"]),
                      "interval_eligible": len(payload["runtime_eligible_cells"]),
                      "eligible_cells": payload["runtime_eligible_cells"]}, indent=2), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--point-audit", type=Path, default=Path("artifacts/multisource_shared_jepa_forecast/multisource_shared_jepa_forecast_audit_20260902.json"))
    parser.add_argument("--mimic-template", default="/private/tmp/mimiciv_observation_transitions_{horizon}h_full_v31_treatment_unit_safe.parquet")
    parser.add_argument("--external-examples", type=Path, default=Path("/private/tmp/nwicu_event_examples_41_grid.parquet"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/multisource_interval_residuals_20260902"))
    parser.add_argument("--output", type=Path, default=Path("models/multisource_interval_calibrator.json"))
    parser.add_argument("--findings", type=Path, default=Path("MULTISOURCE_INTERVAL_CALIBRATION_FINDINGS_20260902.md"))
    parser.add_argument("--selection-max-rows", type=int, default=15000)
    parser.add_argument("--max-rows", type=int, default=30000)
    parser.add_argument("--max-external-rows", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
