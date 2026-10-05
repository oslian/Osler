"""Isolate upstream input-support drift without fitting another selector."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from scripts.audit_amplitude_calibration import predict, evaluate
from scripts.audit_forecast_aware_routing import fixed_predictions
from scripts.audit_preanchor_context_routing import _mimic_context
from scripts.audit_state_conditioned_kidney import (
    read_cohort, source_roles, prepare_external, encode_features, positions, comparison_gate,
)
from whole_body_jepa.input_support import learn_input_support, apply_input_support
from whole_body_jepa.multisource_intervals import sha256
from whole_body_jepa.runtime import load_release


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    previous = json.loads((args.previous / "report.json").read_text())
    for filename, digest in previous["release_binding"].items():
        if sha256(args.model_dir / filename) != digest:
            raise ValueError(f"release changed: {filename}")
    for path, key in ((args.experts / "candidate_heads.joblib", "expert_bundle_sha256"),
                      (args.binary / "selector_heads.joblib", "binary_bundle_sha256")):
        if sha256(path) != previous[key]:
            raise ValueError(f"frozen reference changed: {path}")
    assets = load_release(args.model_dir)
    print("audit source support using development inputs only", flush=True)
    source = read_cohort(args.eicu, assets, "eicu", args.horizon)
    train = source_roles(source)["train"]
    contract = learn_input_support(source.loc[train], assets.checkpoint["variables"],
                                  max_age_hours=assets.max_age_hours)
    all_support = learn_input_support(source, assets.checkpoint["variables"], max_age_hours=assets.max_age_hours)
    if contract["supported"] != all_support["supported"]:
        raise ValueError("source masking changes held-out support; existing calibration cannot be reused")
    report = {"schema": "input_support_parity.v1", "source_support": contract,
              "source_rows": len(source), "source_subjects": int(source.subject_id.nunique()),
              "source_population": "AKI-selected, onset-anchored",
              "target_population": "whole-ICU, admission-anchored",
              "source_equivalence_verified": True, "release_binding": previous["release_binding"],
              "protocol_sha256": sha256(Path("INPUT_SUPPORT_PARITY_PROTOCOL_20260905.md")),
              "previous_report_sha256": sha256(args.previous / "report.json"),
              "amplitude_bundle_sha256": sha256(args.previous / "amplitude_heads.joblib"),
              "expert_bundle_sha256": previous["expert_bundle_sha256"],
              "binary_bundle_sha256": previous["binary_bundle_sha256"],
              "source_fitting_performed": False, "mimic_fitting_performed": False,
              "external_is_untouched_lockbox": False, "runtime_modified": False,
              "promotion_allowed": False, "causal_claim_allowed": False, "targets": {}}
    (args.output_dir / "input_support.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"supported": contract["supported"], "unsupported": contract["unsupported"]}), flush=True)
    del source
    old = joblib.load(args.experts / "candidate_heads.joblib")
    binary = joblib.load(args.binary / "selector_heads.joblib")
    amplitude = joblib.load(args.previous / "amplitude_heads.joblib")
    print("encode original MIMIC panel and reproduce previous evaluation", flush=True)
    frame, rows, features, lookup = prepare_external(args, assets)
    report["target_support"] = learn_input_support(frame, assets.checkpoint["variables"],
                                                  max_age_hours=assets.max_age_hours)
    print("extract unchanged MIMIC pre-anchor care context", flush=True)
    context, _ = _mimic_context(frame, lookup, args.mimic_root)
    reference_points = {}
    def infer(target, which, representation):
        sub, pos = frame.iloc[which].reset_index(drop=True), positions(which, lookup)
        base, ctx = representation[target][pos], context[pos]
        pred = fixed_predictions(old[target], sub, base, ctx, target, args.seed)
        alpha = predict(amplitude[target]["heads"], old[target], binary[target], base, ctx, pred, sub, args.seed)
        result = evaluate(sub, pred, alpha, amplitude[target]["scales"], target, "mimic")
        point = {m: sub[f"{target}_t"].to_numpy(float) + a * pred["expert"] for m, a in alpha.items()}
        return result, point, sub
    for target, which in rows.items():
        result, points, _ = infer(target, which, features)
        for method, item in result["methods"].items():
            expected = previous["targets"][target]["evaluation"]["mimic"]["methods"][method]["mae"]
            if not np.isclose(item["mae"], expected, atol=1e-8, rtol=0):
                raise ValueError(f"original point reproduction failed: {target}/{method}")
        report["targets"][target] = {"original": result}
        reference_points[target] = points
    del features
    print("re-encode complete MIMIC histories using source input support", flush=True)
    aligned = apply_input_support(frame, contract)
    union = np.asarray([r for r, _ in sorted(lookup.items(), key=lambda item: item[1])])
    features = encode_features(aligned, union, assets, args.horizon)
    del aligned
    for target, which in rows.items():
        result, points, sub = infer(target, which, features)
        truth = sub[f"future_{target}"].to_numpy(float)
        item = report["targets"][target]
        item["aligned"] = result
        item["paired_input_effect"] = {m: comparison_gate(sub, points[m], reference_points[target][m], truth,
            f"input_support:{target}:{m}") for m in ("conditional", "binary", "always_expert")}
        item["source_gate_pass"] = all(e["methods"]["conditional"]["full_gate_pass"]
            for k, e in previous["targets"][target]["evaluation"].items() if k != "mimic")
        item["all_role_gate_pass"] = bool(item["source_gate_pass"]
            and result["methods"]["conditional"]["full_gate_pass"]
            and item["paired_input_effect"]["conditional"]["pass"]
            and item["paired_input_effect"]["conditional"]["all"]["pass"])
        print(json.dumps({"target": target,
            "original": {m: item["original"]["methods"][m]["mae"] for m in ("conditional", "always_expert", "persistence")},
            "aligned": {m: result["methods"][m]["mae"] for m in ("conditional", "always_expert", "persistence")},
            "all_role_gate_pass": item["all_role_gate_pass"]}), flush=True)
    report["gate_pass_targets"] = [t for t, i in report["targets"].items() if i["all_role_gate_pass"]]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("FINISHED " + json.dumps(report["gate_pass_targets"]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, default=Path("artifacts/amplitude_calibration_20260905"))
    p.add_argument("--experts", type=Path, default=Path("artifacts/expected_error_routing_20260905_v2"))
    p.add_argument("--binary", type=Path, default=Path("artifacts/forecast_aware_routing_20260905"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/input_support_parity_20260905"))
    p.add_argument("--eicu", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/eicu_future_label_repaired/eicu_aki_transitions_6h.parquet"))
    p.add_argument("--mimic", type=Path, default=Path("/Users/chunyouchang/Desktop/llm_project/medical_jepa/mimiciv_observation_transitions_6h_full_v31_treatment_unit_safe.parquet"))
    p.add_argument("--mimic-root", type=Path, default=Path("/Users/chunyouchang/Downloads/mimic-iv-3.1"))
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260904)
    run(p.parse_args())


if __name__ == "__main__":
    main()
