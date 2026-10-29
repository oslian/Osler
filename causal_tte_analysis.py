"""Target-trial emulation by inverse probability weighting: design diagnostics, the one-shot estimate, and review.

A JSON spec names the design and outcome tables, key, subject, treatment, the
frozen adjustment set, the outcomes (change from baseline), the negative-control
outcome and the directional benchmark, so each emulation is configuration.

``design``   reads only the design table (no outcome column may be present):
             propensity model (logistic regression on the adjustment set),
             stabilized weights truncated at the 1st/99th percentiles (Cole &
             Hernán 2008), overlap, weighted standardized mean differences
             (Austin & Stuart 2015) and Kish effective sample size per arm.
``estimate`` runs once per protocol (``osler_jepa.frozen_protocol``): the spec
             and design table come from the frozen protocol, the outcome table
             must be the one recorded by the outcome-extraction step, and the
             step is claimed atomically.
``review``   checks a separate review record naming the exact protocol and
             result digests; only an explicit approval of both permits a
             causal claim.

Estimand (ICH E9(R1) hypothetical strategy): the average treatment effect, in
eligible stays with the outcome's baseline value, on the outcome at time zero +
24 h in a hypothetical world where neither death nor hospital discharge occurred
before time zero + 24 h and everyone was tested.  Operational rule: an outcome
is observed only for a stay with none of the spec's ``intercurrent_events``
(death or discharge before + 24 h) and a sample in the 18-30 h window; that
sample stands for the 24 h value (consistency).  Identification additionally
assumes, given the baseline adjustment set and the arm, that intercurrent
events and missed testing are independent of the outcome (no evolving
post-treatment cause of both) and that every stratum has a positive
probability of being observed (observation positivity).  Weights: stabilized inverse probability of treatment x inverse
probability of outcome measurement (Robins, Hernán & Brumback 2000), the product
truncated at its 1st/99th percentiles.  Subject-cluster bootstrap CIs redo the
imputation and refit both models in every replicate; failing replicates count
as invalid.  The observed-survivors analysis (no recorded death before time
zero + 24 h) is a reported descriptive subset association.

Readiness rules (all on unrounded values): design gates (max |weighted SMD| <=
0.10, >= 95% of stays with propensity in [0.05, 0.95], effective sample size >=
500 per arm); for the primary and negative-control analyses, final-weight
diagnostics on the measured stays (max |SMD| between arms <= 0.10, max |SMD| of
each weighted arm against its whole analysis population <= 0.10, effective
sample size >= 500 per arm; observation support: in every covariate stratum
(numeric quintiles, category levels) with >= 50 stays of an arm, >= 5% of that
arm observed); >= 95% valid bootstrap replicates for the primary and
negative-control analyses;
negative-control 95% CI inside +-margin (an equivalence falsification test: it
does not bound confounding of the primary outcome); directional benchmark:
primary 95% CI entirely beyond -margin (an investigator-chosen compatibility
check, not validation of identification).  Secondary and sensitivity analyses
are reported, not gated; any analysis that cannot be computed is reported as a
structured failure (a failing primary or negative control fails readiness).
The observed-survivors analysis is a descriptive subset association, not a
causal while-alive effect.  Research-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from osler_jepa.frozen_protocol import claim_step, load_frozen, record_step, reviewed, sha256, step_record, write_once
from osler_jepa.instrumental_variable import cluster_bootstrap


MAX_SMD = 0.10
OVERLAP_BOUNDS = (0.05, 0.95)
MIN_OVERLAP_SHARE = 0.95
MIN_ESS = 500.0
MIN_VALID_BOOTSTRAP_SHARE = 0.95
MIN_STRATUM_STAYS = 50
MIN_STRATUM_OBSERVED = 0.05
SMD_TOLERANCE = 1e-9
TRUNCATION_PERCENTILES = (1, 99)
OUTCOME_PREFIXES = ("day1_", "nc_", "fu_")


def adjustment_frame(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Frozen adjustment set: numeric covariates median-imputed with missing indicators, categorical one-hot."""

    columns = {}
    for name in spec["numeric_covariates"]:
        values = pd.to_numeric(frame[name], errors="coerce")
        columns[name] = values.fillna(values.median())
        if values.isna().any():
            columns[f"{name}_missing"] = values.isna().astype(float)
    numeric = pd.DataFrame(columns, index=frame.index)
    categorical = [pd.get_dummies(frame[name].fillna("unknown").astype(str), prefix=name, dtype=float)
                   for name in spec["categorical_covariates"]]
    return pd.concat([numeric, *categorical], axis=1)


def fitted_probability(x: np.ndarray, label: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return model.fit(x, label).predict_proba(x)[:, 1]


def truncated(weights: np.ndarray) -> np.ndarray:
    low, high = np.percentile(weights, TRUNCATION_PERCENTILES)
    return np.clip(weights, low, high)


def stabilized_weights(propensity: np.ndarray, treated: np.ndarray) -> np.ndarray:
    share = treated.mean()
    return np.where(treated == 1, share / propensity, (1 - share) / (1 - propensity))


def weighted_smd(values: np.ndarray, group: np.ndarray, weights: np.ndarray) -> float:
    def moments(mask):
        mean = np.average(values[mask], weights=weights[mask])
        return mean, np.average((values[mask] - mean) ** 2, weights=weights[mask])
    (m1, v1), (m0, v0) = moments(group == 1), moments(group == 0)
    scale = max(1.0, abs(m1), abs(m0))
    pooled = np.sqrt((v1 + v0) / 2.0)
    if pooled > SMD_TOLERANCE * scale:
        return float((m1 - m0) / pooled)
    return 0.0 if abs(m1 - m0) <= SMD_TOLERANCE * scale else float("inf")   # constants: equal, or complete separation


def max_abs_smd(covariates: pd.DataFrame, group: np.ndarray, weights: np.ndarray) -> float:
    return max(abs(weighted_smd(covariates[name].to_numpy(), group, weights)) for name in covariates)


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / (weights ** 2).sum())


def design(spec: dict) -> dict:
    frame = pd.read_parquet(spec["design_table"])
    assert not any(c.startswith(OUTCOME_PREFIXES) for c in frame.columns), "design table must not contain outcomes"
    covariates = adjustment_frame(frame, spec)
    x = covariates.to_numpy()
    treated = frame[spec["treatment"]].to_numpy(dtype=int)
    propensity = fitted_probability(x, treated)
    weights = truncated(stabilized_weights(propensity, treated))
    ones = np.ones(len(frame))
    unweighted = {name: weighted_smd(covariates[name].to_numpy(), treated, ones) for name in covariates}
    weighted = {name: weighted_smd(covariates[name].to_numpy(), treated, weights) for name in covariates}
    overlap = float(((propensity >= OVERLAP_BOUNDS[0]) & (propensity <= OVERLAP_BOUNDS[1])).mean())
    ess = {"treated": effective_sample_size(weights[treated == 1]), "control": effective_sample_size(weights[treated == 0])}
    max_weighted = max(abs(v) for v in weighted.values())
    checks = {"covariate_balance": bool(max_weighted <= MAX_SMD),
              "overlap": overlap >= MIN_OVERLAP_SHARE,
              "effective_sample_size": min(ess.values()) >= MIN_ESS}
    return {
        "stays": int(len(frame)), "subjects": int(frame[spec["subject"]].nunique()),
        "treated_share": round(float(treated.mean()), 4), "adjustment_columns": int(x.shape[1]),
        "propensity_quantiles": {arm: np.quantile(propensity[treated == value], [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]).round(3).tolist()
                                 for arm, value in (("treated", 1), ("control", 0))},
        "overlap_share": round(overlap, 4),
        "effective_sample_size": {k: round(v, 1) for k, v in ess.items()},
        "max_abs_smd": {"unweighted": round(max(abs(v) for v in unweighted.values()), 4), "weighted": round(max_weighted, 4)},
        "smd_unweighted": {k: round(v, 3) for k, v in unweighted.items()},
        "smd_weighted": {k: round(v, 3) for k, v in weighted.items()},
        "design_checks": checks, "design_passes": all(checks.values()),
        "outcomes_read": False,
    }


def outcome_weights(x: np.ndarray, treated: np.ndarray, measured: np.ndarray, *, truncate: bool = True,
                    measurement_weighting: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Weights of measured stays (IPT x inverse measurement probability) and the measurement probabilities."""

    treatment = stabilized_weights(fitted_probability(x, treated), treated)
    probability = (fitted_probability(np.column_stack([x, treated]), measured.astype(int))
                   if measurement_weighting and not measured.all() else np.ones(len(treated)))
    weights = treatment[measured] / probability[measured]
    return (truncated(weights) if truncate else weights), probability


def weighted_effect(frame: pd.DataFrame, spec: dict, outcome: np.ndarray, **options) -> float:
    """Effect on one (re)sample: imputation, both models and the weighted mean difference; nan if fitting fails."""

    try:
        x = adjustment_frame(frame, spec).to_numpy()
        treated = frame[spec["treatment"]].to_numpy(dtype=int)
        measured = np.isfinite(outcome)
        weights, _ = outcome_weights(x, treated, measured, **options)
        y, a = outcome[measured], treated[measured]
        return float(np.average(y[a == 1], weights=weights[a == 1]) - np.average(y[a == 0], weights=weights[a == 0]))
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return float("nan")


def final_weight_diagnostics(covariates: pd.DataFrame, treated: np.ndarray, measured: np.ndarray, weights: np.ndarray) -> dict:
    """Balance of the weighted measured stays between arms and against each arm's whole analysis population."""

    measured_covariates = covariates[measured].reset_index(drop=True)
    arms = treated[measured]
    against_population = {}
    for arm, value in (("treated", 1), ("control", 0)):
        sample = measured_covariates[arms == value]
        stacked = pd.concat([covariates, sample], ignore_index=True)
        group = np.r_[np.zeros(len(covariates)), np.ones(len(sample))]
        against_population[arm] = max_abs_smd(stacked, group, np.r_[np.ones(len(covariates)), weights[arms == value]])
    return {"max_abs_smd_between_arms": max_abs_smd(measured_covariates, arms, weights),
            "max_abs_smd_against_population": against_population,
            "weight_ess": {"treated": effective_sample_size(weights[arms == 1]), "control": effective_sample_size(weights[arms == 0])},
            "max_weight": float(weights.max())}


def observation_support(covariates: pd.DataFrame, treated: np.ndarray, measured: np.ndarray) -> dict:
    """Lowest observed share over covariate strata (numeric quintiles, category levels) with enough stays of an arm."""

    lowest = {"share": 1.0, "stratum": None}
    for name in covariates:
        values = covariates[name]
        strata = values if values.nunique() <= 10 else pd.qcut(values, 5, labels=False, duplicates="drop")
        for arm in (0, 1):
            in_arm = treated == arm
            table = pd.DataFrame({"stratum": strata[in_arm].to_numpy(), "observed": measured[in_arm]}).groupby("stratum")["observed"]
            shares = table.mean()[table.size() >= MIN_STRATUM_STAYS]
            if len(shares) and shares.min() < lowest["share"]:
                lowest = {"share": float(shares.min()), "stratum": f"{name}={shares.idxmin()} arm={arm}"}
    return lowest


def effect(frame: pd.DataFrame, spec: dict, outcome: np.ndarray, **options) -> dict:
    """Weighted effect with diagnostics and bootstrap CI; a structured failure if it cannot be computed."""

    try:
        return _effect(frame, spec, outcome, **options)
    except (ValueError, IndexError, ZeroDivisionError, np.linalg.LinAlgError) as error:
        return {"stays": int(len(frame)), "failed": f"{type(error).__name__}: {error}"}


def _effect(frame: pd.DataFrame, spec: dict, outcome: np.ndarray, **options) -> dict:
    covariates = adjustment_frame(frame, spec)
    treated = frame[spec["treatment"]].to_numpy(dtype=int)
    measured = np.isfinite(outcome)
    weights, probability = outcome_weights(covariates.to_numpy(), treated, measured, **options)
    replicates = spec["bootstrap_replicates"]
    result = cluster_bootstrap(lambda r: weighted_effect(frame.iloc[r].reset_index(drop=True), spec, outcome[r], **options),
                               frame[spec["subject"]].to_numpy(), replicates=replicates, seed=0)
    return {"stays": int(len(frame)),
            "measured_share": {"treated": float(measured[treated == 1].mean()), "control": float(measured[treated == 0].mean())},
            "min_measurement_probability": float(probability.min()),
            **final_weight_diagnostics(covariates, treated, measured, weights),
            "observation_support": observation_support(covariates, treated, measured),
            "estimate": result.estimate, "ci95": [result.low, result.high],
            "valid_bootstrap_share": result.replicates / replicates}


def diagnostics_pass(result: dict) -> bool:
    return ("failed" not in result and result["max_abs_smd_between_arms"] <= MAX_SMD
            and max(result["max_abs_smd_against_population"].values()) <= MAX_SMD
            and min(result["weight_ess"].values()) >= MIN_ESS and result["observation_support"]["share"] >= MIN_STRATUM_OBSERVED)


def rounded(value):
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return round(value, 4) if isinstance(value, float) else value


def estimate(protocol_path: Path) -> tuple[dict, Path]:
    protocol, spec = load_frozen(protocol_path)
    extraction = step_record(protocol_path, "extract_outcomes")
    expected = {protocol["spec"]: sha256(Path(protocol["spec"])), spec["design_table"]: sha256(Path(spec["design_table"]))}
    outputs = extraction["outputs_sha256"]
    if (extraction["inputs_sha256"] != expected or spec["outcome_table"] not in outputs
            or any(sha256(Path(name)) != digest for name, digest in outputs.items())):
        raise RuntimeError("outcome table was not produced by the outcome extraction under this protocol")
    output = write_once(Path(spec["estimate_output"]))
    claim_step(protocol_path, "estimate")
    design_frame = pd.read_parquet(spec["design_table"])
    outcome_frame = pd.read_parquet(spec["outcome_table"])
    for table in (design_frame, outcome_frame):
        if table[spec["key"]].isna().any() or not table[spec["key"]].is_unique:
            raise RuntimeError("missing or duplicate keys")
    frame = design_frame.merge(outcome_frame, on=spec["key"], how="left", validate="one_to_one")
    treated = frame[spec["treatment"]].to_numpy(dtype=int)

    def change(rows, name):
        """Outcome change, unobserved for a stay with an intercurrent event (hypothetical strategy)."""
        intercurrent = rows[spec["intercurrent_events"]].fillna(0).to_numpy().any(axis=1)
        return np.where(intercurrent, np.nan, (rows[f"day1_{name}"] - rows[f"baseline_{name}"]).to_numpy(dtype=float))

    def population(name, rows=frame):
        return rows[rows[f"baseline_{name}"].notna()].reset_index(drop=True)

    results = {f"{name}_change": effect(population(name), spec, change(population(name), name)) for name in spec["outcomes"]}
    negative = spec["negative_control"]
    # complete-case falsification test: having the earlier sample is pre-treatment, so no measurement weighting is needed
    with_control = frame[frame[negative["column"]].notna()].reset_index(drop=True)
    control = results[f"negative_control_{negative['column']}"] = effect(with_control, spec, with_control[negative["column"]].to_numpy(dtype=float))
    primary = spec["primary_outcome"]
    rows = population(primary)
    survivors = population(primary, frame[frame[spec["death_event"]] == 0])
    sensitivity = {"untruncated_weights": effect(rows, spec, change(rows, primary), truncate=False),
                   "no_measurement_weighting": effect(rows, spec, change(rows, primary), measurement_weighting=False),
                   "observed_survivors_subset_association": effect(survivors, spec, change(survivors, primary))}
    follow_up = {column: {"treated": float(frame.loc[treated == 1, column].mean()), "control": float(frame.loc[treated == 0, column].mean())}
                 for column in frame.columns if column.startswith("fu_")}
    main = results[f"{primary}_change"]
    margin = negative["margin"]
    computed = "failed" not in main and "failed" not in control
    checks = {
        **design(spec)["design_checks"],
        "gated_analyses_computed": computed,
        "final_weight_diagnostics": diagnostics_pass(main) and diagnostics_pass(control),
        "bootstrap_valid": computed and min(main["valid_bootstrap_share"], control["valid_bootstrap_share"]) >= MIN_VALID_BOOTSTRAP_SHARE,
        "negative_control_equivalent": computed and bool(-margin <= control["ci95"][0] and control["ci95"][1] <= margin),
        "directional_benchmark": computed and bool(main["ci95"][1] < -margin if spec["benchmark_direction"] == "negative"
                                                   else main["ci95"][0] > margin),
    }
    passes = all(checks.values())
    report = rounded({
        "schema": "causal_tte_analysis.v3", "spec": spec, "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "results": results, "sensitivity": sensitivity, "follow_up_events_by_arm": follow_up,
        "readiness_checks": checks,
        "readiness_status": "ready_for_external_review" if passes else "failed",
        "causal_claim_allowed": False,
        "clinical_claim_allowed": False,
        "runtime_decision_authority": False,
    })
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    record_step(protocol_path, "estimate", [protocol["spec"], spec["design_table"], spec["outcome_table"]], [str(output)])
    return report, output


def review(protocol_path: Path, review_path: Path) -> dict:
    """Causal permission for an estimate: the completed estimate step of this protocol, an explicit approval of
    exactly this protocol and result, and a passing result."""

    protocol, spec = load_frozen(protocol_path)
    result_path = Path(spec["estimate_output"])
    receipt = step_record(protocol_path, "estimate")
    inputs = {name: sha256(Path(name)) for name in (protocol["spec"], spec["design_table"], spec["outcome_table"])}
    if receipt["inputs_sha256"] != inputs or receipt["outputs_sha256"] != {spec["estimate_output"]: sha256(result_path)}:
        raise RuntimeError("estimate result was not produced by this protocol's estimate step")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["protocol_sha256"] != sha256(protocol_path):
        raise RuntimeError("estimate result names another protocol")
    approved = reviewed(review_path, protocol_path, result_path)
    return {"protocol_sha256": sha256(protocol_path), "result_sha256": sha256(result_path), "review": str(review_path),
            "review_sha256": sha256(review_path), "approved": approved,
            "causal_claim_allowed": bool(approved and result["readiness_status"] == "ready_for_external_review"),
            "clinical_claim_allowed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("design", "estimate", "review"))
    parser.add_argument("--spec", type=Path, help="design only (estimate and review read the spec from the protocol)")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--review-record", type=Path)
    parser.add_argument("--output", type=Path, help="design and review outputs")
    args = parser.parse_args()
    if args.command == "design":
        report = design(json.loads(args.spec.read_text(encoding="utf-8")))
        args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    elif args.command == "estimate":
        report, _ = estimate(args.protocol)
    else:
        report = review(args.protocol, args.review_record)
        write_once(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    shown = {k: v for k, v in report.items() if not k.startswith(("smd_", "spec"))}
    print(json.dumps(shown, indent=1, default=str))


if __name__ == "__main__":
    main()
