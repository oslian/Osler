"""Generic preference-instrument analysis: design diagnostics and the one-shot estimate.

A JSON spec names the design table, key, cluster and instrument columns, the
frozen adjustment set, the outcomes (change from baseline) and the benchmark,
so each new target-trial emulation is configuration, not new code.

``design``   reads only the design table: instrument strength, uptake by
             instrument quintile, baseline balance across instrument halves.
``estimate`` runs only under a frozen protocol (every hashed file unchanged),
             once: 2SLS with cluster-bootstrap CIs for the primary outcome,
             secondary outcomes, a negative control and a pre-treatment placebo.

Readiness rules (same as the eICU analysis, ``crystalloid_iv_estimate.py``):
F >= 10; |SMD| <= 0.10 across instrument halves; outcome availability gap <=
0.05; benchmark estimate inside the spec's range with the CI on the expected
side of 0; placebo and negative-control CIs contain 0; external review needed
before ``causal_claim_allowed``.  Research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_value_conditional_interval_audit import verify_protocol
from osler_jepa.instrumental_variable import cluster_bootstrap, first_stage_f, standardized_mean_difference, two_stage_least_squares


MAX_SMD = 0.10
MAX_AVAILABILITY_GAP = 0.05
MIN_F = 10.0


def covariate_matrix(frame: pd.DataFrame, spec: dict, exclude: tuple[str, ...] = ()) -> np.ndarray:
    columns = []
    for name in spec["numeric_covariates"]:
        if name in exclude:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        columns.append(values.fillna(values.median()).to_numpy())
        if values.isna().any():
            columns.append(values.isna().astype(float).to_numpy())
    for name in spec["categorical_covariates"]:
        dummies = pd.get_dummies(frame[name].fillna("unknown").astype(str), prefix=name, drop_first=True, dtype=float)
        columns.extend(dummies.to_numpy().T)
    return np.column_stack(columns)


def design(spec: dict) -> dict:
    frame = pd.read_parquet(spec["design_table"])
    assert not any(c.startswith("day1_") for c in frame.columns), "design table must not contain outcomes"
    x = covariate_matrix(frame, spec)
    d = frame[spec["treatment"]].to_numpy(dtype=float)
    z = frame[spec["instrument"]].to_numpy(dtype=float)
    high = z > np.median(z)
    quintile = pd.qcut(z, 5, labels=False, duplicates="drop")
    return {
        "stays": int(len(frame)), "clusters": int(frame[spec["cluster"]].nunique()),
        "treated_share": round(float(d.mean()), 4),
        "treated_share_by_cluster": frame.groupby(spec["cluster"])[spec["treatment"]].mean().round(3).to_dict(),
        "first_stage_f_unadjusted": round(first_stage_f(d, z), 1),
        "first_stage_f_adjusted": round(first_stage_f(d, z, x), 1),
        "uptake_by_instrument_quintile": pd.Series(d).groupby(quintile).mean().round(3).tolist(),
        "balance_high_vs_low_instrument_smd": {name: round(standardized_mean_difference(pd.to_numeric(frame[name], errors="coerce"), high), 3)
                                               for name in spec["numeric_covariates"]},
        "balance_treated_vs_untreated_smd": {name: round(standardized_mean_difference(pd.to_numeric(frame[name], errors="coerce"), d.astype(bool)), 3)
                                             for name in spec["numeric_covariates"]},
        "outcomes_read": False,
    }


def iv_estimate(frame: pd.DataFrame, spec: dict, outcome: np.ndarray, *, cluster: str, exclude: tuple[str, ...] = ()) -> dict:
    keep = np.isfinite(outcome)
    rows = frame[keep].reset_index(drop=True)
    y = outcome[keep]
    x = covariate_matrix(rows, spec, exclude)
    d = rows[spec["treatment"]].to_numpy(dtype=float)
    z = rows[spec["instrument"]].to_numpy(dtype=float)
    result = cluster_bootstrap(lambda r: two_stage_least_squares(y[r], d[r], z[r], x[r]),
                               rows[cluster].to_numpy(), replicates=spec["bootstrap_replicates"], seed=0)
    naive = np.linalg.lstsq(np.column_stack([d, np.ones_like(d), x]), y, rcond=None)[0][0]
    return {"stays": int(len(rows)), "clusters": int(rows[cluster].nunique()),
            "iv_estimate": round(result.estimate, 3), "ci95": [round(result.low, 3), round(result.high, 3)],
            "adjusted_ols_estimate": round(float(naive), 3), "first_stage_f": round(first_stage_f(d, z, x), 1)}


def contains_zero(result: dict) -> bool:
    return result["ci95"][0] <= 0 <= result["ci95"][1]


def estimate(spec: dict, protocol_path: Path) -> dict:
    protocol = verify_protocol(protocol_path)
    frame = pd.read_parquet(spec["design_table"]).merge(pd.read_parquet(spec["outcome_table"]), on=spec["key"], how="left")
    change = {name: (frame[f"day1_{name}"] - frame[f"baseline_{name}"]).to_numpy(dtype=float) for name in spec["outcomes"]}
    primary, negative = spec["primary_outcome"], spec["negative_control_outcome"]
    results = {f"{name}_change": iv_estimate(frame, spec, change[name], cluster=spec["cluster"]) for name in spec["outcomes"]}
    results["placebo_baseline_" + primary] = iv_estimate(frame, spec, frame[f"baseline_{primary}"].to_numpy(dtype=float),
                                                         cluster=spec["cluster"], exclude=(f"baseline_{primary}",))
    sensitivity = {f"{primary}_change_{spec['sensitivity_cluster']}_bootstrap":
                   iv_estimate(frame, spec, change[primary], cluster=spec["sensitivity_cluster"])}
    z = frame[spec["instrument"]]
    high = (z > z.median()).to_numpy()
    available = np.isfinite(change[primary])
    balance = {name: standardized_mean_difference(pd.to_numeric(frame[name], errors="coerce"), high) for name in spec["numeric_covariates"]}
    main = results[f"{primary}_change"]
    low, high_bound = spec["benchmark_range"]
    expected_negative = high_bound < 0
    checks = {
        "instrument_strength": main["first_stage_f"] >= MIN_F,
        "covariate_balance": bool(max(abs(v) for v in balance.values()) <= MAX_SMD),
        "outcome_availability": bool(abs(available[high].mean() - available[~high].mean()) <= MAX_AVAILABILITY_GAP),
        "benchmark_reproduced": bool(low <= main["iv_estimate"] <= high_bound
                                     and (main["ci95"][1] < 0 if expected_negative else main["ci95"][0] > 0)),
        "placebo_null": contains_zero(results["placebo_baseline_" + primary]),
        "negative_control_null": contains_zero(results[f"{negative}_change"]),
    }
    review = protocol.get("external_review")
    passes = all(checks.values())
    return {
        "schema": "causal_iv_analysis.v1", "spec": spec, "protocol": str(protocol_path),
        "results": results, "sensitivity": sensitivity,
        "balance_high_vs_low_instrument_smd": {k: round(v, 3) for k, v in balance.items()},
        "outcome_availability": {"high_instrument": round(float(available[high].mean()), 4), "low_instrument": round(float(available[~high].mean()), 4)},
        "readiness_checks": checks,
        "readiness_status": "causal_estimate_reviewed" if passes and review else "ready_for_external_review" if passes else "failed",
        "external_review": review,
        "causal_claim_allowed": bool(passes and review),
        "clinical_claim_allowed": False,
        "runtime_decision_authority": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("design", "estimate"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.command == "design":
        report = design(spec)
    else:
        if args.protocol is None:
            raise ValueError("estimate requires --protocol")
        report = estimate(spec, args.protocol)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    shown = report if args.command == "design" else {k: report[k] for k in ("results", "sensitivity", "readiness_checks", "readiness_status", "causal_claim_allowed")}
    print(json.dumps(shown, indent=1, default=str))


if __name__ == "__main__":
    main()
