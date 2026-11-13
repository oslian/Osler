"""One-shot estimate: balanced crystalloids versus 0.9% saline on 24-hour chloride (hospital-preference IV).

Runs only under the frozen protocol (``--protocol``; every hashed file must be
unchanged).  Estimator: 2SLS of the outcome change on the predominantly-balanced
strategy, instrumented by the leave-one-out hospital preference, adjusted for
the frozen covariate set (``crystalloid_iv_design.covariate_matrix``); 95% CI by
hospital cluster bootstrap (1,000 replicates).

Readiness rules (frozen in the protocol before outcomes were extracted):

1. instrument strength: adjusted first-stage F >= 10;
2. balance: |SMD| <= 0.10 for every patient covariate between high- and
   low-preference hospitals (above vs below the median instrument);
3. outcome availability: the share of stays with a day-1 chloride differs by
   <= 0.05 between high- and low-preference hospitals;
4. benchmark (randomized trials: balanced crystalloids raise chloride less
   than saline): chloride-change estimate in [-8, -0.3] mmol/L with the CI
   upper bound < 0;
5. placebo: the IV estimate on baseline (pre-treatment) chloride, adjusting
   for everything else, has a CI containing 0;
6. negative control: the IV estimate on 24-hour platelet change has a CI
   containing 0;
7. external review: required for a non-randomized design; until a review
   record exists the result is ``ready_for_external_review`` at best and
   ``causal_claim_allowed`` stays false.

Bicarbonate change is a secondary outcome without a benchmark (randomized
trials show no consistent bicarbonate difference).  Research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crystalloid_iv_design import PATIENT_COVARIATES, covariate_matrix
from external_value_conditional_interval_audit import verify_protocol
from osler_jepa.instrumental_variable import cluster_bootstrap, first_stage_f, standardized_mean_difference, two_stage_least_squares


BENCHMARK_RANGE = (-8.0, -0.3)
MAX_SMD = 0.10
MAX_AVAILABILITY_GAP = 0.05
MIN_F = 10.0
REPLICATES = 1000


def iv_estimate(frame: pd.DataFrame, outcome: np.ndarray, exclude: tuple[str, ...] = ()) -> dict:
    keep = np.isfinite(outcome)
    rows = frame[keep].reset_index(drop=True)
    y = outcome[keep]
    x, names = covariate_matrix(rows)
    columns = [i for i, name in enumerate(names) if not any(name == e or name.startswith(f"{e}_") for e in exclude)]
    x = x[:, columns]
    d = rows["treated_balanced"].to_numpy(dtype=float)
    z = rows["instrument_hospital_preference"].to_numpy(dtype=float)
    result = cluster_bootstrap(lambda r: two_stage_least_squares(y[r], d[r], z[r], x[r]),
                               rows["hospitalid"].to_numpy(), replicates=REPLICATES, seed=0)
    naive = np.linalg.lstsq(np.column_stack([d, np.ones_like(d), x]), y, rcond=None)[0][0]
    return {"stays": int(len(rows)), "hospitals": int(rows["hospitalid"].nunique()),
            "iv_estimate": round(result.estimate, 3), "ci95": [round(result.low, 3), round(result.high, 3)],
            "adjusted_ols_estimate": round(float(naive), 3), "first_stage_f": round(first_stage_f(d, z, x), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=Path("eicu_crystalloid_iv_design.parquet"))
    parser.add_argument("--outcomes", type=Path, default=Path("eicu_crystalloid_iv_outcomes.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_protocol(args.protocol)
    frame = pd.read_parquet(args.design).merge(pd.read_parquet(args.outcomes), on="patientunitstayid", how="left")
    high = frame["instrument_hospital_preference"] > frame["instrument_hospital_preference"].median()
    chloride = (frame["day1_chloride"] - frame["baseline_chloride"]).to_numpy(dtype=float)
    results = {
        "primary_chloride_change": iv_estimate(frame, chloride),
        "secondary_bicarbonate_change": iv_estimate(frame, (frame["day1_bicarbonate"] - frame["baseline_bicarbonate"]).to_numpy(dtype=float)),
        "negative_control_platelet_change": iv_estimate(frame, (frame["day1_platelets"] - frame["baseline_platelets"]).to_numpy(dtype=float)),
        "placebo_baseline_chloride": iv_estimate(frame, frame["baseline_chloride"].to_numpy(dtype=float), exclude=("baseline_chloride",)),
    }
    available = np.isfinite(chloride)
    balance = {name: standardized_mean_difference(pd.to_numeric(frame[name], errors="coerce"), high)
               for name in PATIENT_COVARIATES if name != "typed_volume_ml"}
    primary = results["primary_chloride_change"]
    checks = {
        "instrument_strength": primary["first_stage_f"] >= MIN_F,
        "covariate_balance": bool(max(abs(v) for v in balance.values()) <= MAX_SMD),
        "outcome_availability": bool(abs(available[high].mean() - available[~high].mean()) <= MAX_AVAILABILITY_GAP),
        "benchmark_reproduced": bool(BENCHMARK_RANGE[0] <= primary["iv_estimate"] <= BENCHMARK_RANGE[1] and primary["ci95"][1] < 0),
        "placebo_null": bool(results["placebo_baseline_chloride"]["ci95"][0] <= 0 <= results["placebo_baseline_chloride"]["ci95"][1]),
        "negative_control_null": bool(results["negative_control_platelet_change"]["ci95"][0] <= 0 <= results["negative_control_platelet_change"]["ci95"][1]),
    }
    review = protocol.get("external_review")
    passes = all(checks.values())
    output = {
        "schema": "crystalloid_iv_estimate.v1",
        "protocol": str(args.protocol),
        "estimand": "effect of a predominantly balanced (vs predominantly 0.9% saline) crystalloid strategy over the first 24 h of ICU on chloride change at ~24 h, among compliers with hospital preference (LATE)",
        "results": results,
        "balance_high_vs_low_preference_smd": {k: round(v, 3) for k, v in balance.items()},
        "outcome_availability": {"high_preference": round(float(available[high].mean()), 4), "low_preference": round(float(available[~high].mean()), 4)},
        "readiness_checks": checks,
        "readiness_status": ("causal_estimate_reviewed" if passes and review else
                             "ready_for_external_review" if passes else "failed"),
        "external_review": review,
        "causal_claim_allowed": bool(passes and review),
        "clinical_claim_allowed": False,
        "runtime_decision_authority": False,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("results", "readiness_checks", "readiness_status", "causal_claim_allowed")}, indent=1))


if __name__ == "__main__":
    main()
