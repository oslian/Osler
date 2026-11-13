import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from causal_tte_analysis import diagnostics_pass, design, effect, estimate, final_weight_diagnostics, observation_support, review, weighted_smd
from osler_jepa import frozen_protocol
from osler_jepa.frozen_protocol import claim_step, load_frozen, record_step, sha256


def synthetic(effect: float, hidden_confounding: float = 0.0, outcome_confounding: float = 0.0) -> None:
    rng = np.random.default_rng(0)
    n = 6000
    severity = rng.normal(size=n)
    hidden = rng.normal(size=n)
    baseline = 105 + 2 * severity + rng.normal(size=n)
    treated = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.8 * severity + (hidden_confounding + outcome_confounding) * hidden)))).astype(int)
    day1 = baseline + effect * treated + 1.5 * severity + (hidden_confounding + outcome_confounding) * 2 * hidden + rng.normal(size=n)
    prior = baseline - 0.5 * severity + hidden_confounding * 2 * hidden + rng.normal(size=n)
    frame = pd.DataFrame({"stay_key": np.arange(n), "subject_id": np.arange(n) // 2, "treated_balanced": treated,
                          "age": rng.normal(60, 10, n), "gender": rng.choice(["F", "M"], n), "severity": severity,
                          "baseline_chloride": baseline})
    frame.to_parquet("design.parquet")
    observed = rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 + severity)))
    died = rng.uniform(size=n) < 0.03
    day1 = np.where(died, 200.0, day1)                        # a sample from a stay with an intercurrent event must not count
    pd.DataFrame({"stay_key": frame["stay_key"], "day1_chloride": np.where(observed, day1, np.nan),
                  "nc_prior_chloride": np.where(rng.uniform(size=n) < 0.6, prior, np.nan),
                  "fu_died_before_24h": died.astype(int), "fu_discharged_alive_before_24h": 0}).to_parquet("outcomes.parquet")
    spec = {"design_table": "design.parquet", "outcome_table": "outcomes.parquet", "estimate_output": "estimate.json",
            "key": "stay_key", "subject": "subject_id", "treatment": "treated_balanced",
            "numeric_covariates": ["age", "severity", "baseline_chloride"], "categorical_covariates": ["gender"],
            "outcomes": ["chloride"], "primary_outcome": "chloride", "negative_control": {"column": "nc_prior_chloride", "margin": 0.5},
            "intercurrent_events": ["fu_died_before_24h", "fu_discharged_alive_before_24h"], "death_event": "fu_died_before_24h",
            "benchmark_direction": "negative", "bootstrap_replicates": 30, "frozen_files": ["design.parquet", "spec.json"]}
    Path("spec.json").write_text(json.dumps(spec))


def freeze_and_extract() -> Path:
    protocol = Path("protocol.json")
    protocol.write_text(json.dumps({"spec": "spec.json", "frozen_sha256": {name: sha256(Path(name)) for name in ("spec.json", "design.parquet")}}))
    claim_step(protocol, "extract_outcomes")
    record_step(protocol, "extract_outcomes", ["spec.json", "design.parquet"], ["outcomes.parquet"])
    return protocol


class CausalTteAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.previous = os.getcwd()
        os.chdir(self.folder.name)
        self.ledger = frozen_protocol.LEDGER
        frozen_protocol.LEDGER = Path(self.folder.name) / "causal_executions"

    def tearDown(self):
        os.chdir(self.previous)
        frozen_protocol.LEDGER = self.ledger
        self.folder.cleanup()

    def run_case(self, effect: float, **confounding) -> dict:
        synthetic(effect, **confounding)
        report = design(json.loads(Path("spec.json").read_text()))
        self.assertTrue(report["design_passes"])
        self.assertGreater(report["max_abs_smd"]["unweighted"], 0.3)
        return estimate(freeze_and_extract())[0]

    def test_known_effect_is_recovered_under_confounding_and_informative_measurement(self):
        report = self.run_case(-3.0)
        self.assertAlmostEqual(report["results"]["chloride_change"]["estimate"], -3.0, delta=0.25)
        self.assertTrue(all(report["readiness_checks"].values()))
        self.assertEqual(report["readiness_status"], "ready_for_external_review")
        self.assertFalse(report["causal_claim_allowed"])

    def test_review_must_approve_exactly_this_protocol_and_result(self):
        self.run_case(-3.0)
        protocol, result = Path("protocol.json"), Path("estimate.json")
        record = {"decision": "approved", "reviewer": "x", "protocol_sha256": sha256(protocol), "result_sha256": sha256(result)}
        Path("review.json").write_text(json.dumps(record))
        self.assertTrue(review(protocol, Path("review.json"))["causal_claim_allowed"])
        Path("review.json").write_text(json.dumps({**record, "result_sha256": "0" * 64}))
        self.assertFalse(review(protocol, Path("review.json"))["causal_claim_allowed"])
        Path("review.json").write_text(json.dumps({**record, "decision": "rejected"}))
        self.assertFalse(review(protocol, Path("review.json"))["causal_claim_allowed"])

    def test_effect_within_the_margin_fails_the_benchmark(self):
        self.assertFalse(self.run_case(-0.3)["readiness_checks"]["directional_benchmark"])

    def test_confounding_shared_with_earlier_chloride_fails_the_negative_control(self):
        report = self.run_case(-3.0, hidden_confounding=1.0)
        self.assertFalse(report["readiness_checks"]["negative_control_equivalent"])
        self.assertEqual(report["readiness_status"], "failed")

    def test_negative_control_cannot_see_confounding_of_later_chloride_only(self):
        # documents the limit Codex demonstrated: a passing control does not bound primary-outcome confounding
        report = self.run_case(0.0, outcome_confounding=1.0)
        self.assertTrue(report["readiness_checks"]["negative_control_equivalent"])
        self.assertGreater(report["results"]["chloride_change"]["ci95"][0], 0.5)     # spurious effect, true effect 0

    def test_outcome_table_must_come_from_the_recorded_extraction(self):
        synthetic(-3.0)
        protocol = freeze_and_extract()
        pd.read_parquet("outcomes.parquet").assign(day1_chloride=100.0).to_parquet("outcomes.parquet")
        with self.assertRaises(RuntimeError):
            estimate(protocol)

    def test_estimate_runs_once_per_protocol(self):
        self.run_case(-3.0)
        Path("estimate.json").unlink()
        with self.assertRaises(FileExistsError):
            estimate(Path("protocol.json"))

    def test_protocol_must_hash_its_spec_and_every_required_file(self):
        synthetic(-3.0)
        Path("protocol.json").write_text(json.dumps({"spec": "spec.json", "frozen_sha256": {"spec.json": sha256(Path("spec.json"))}}))
        with self.assertRaises(RuntimeError):
            load_frozen(Path("protocol.json"))
        Path("protocol.json").write_text(json.dumps({"spec": "spec.json", "frozen_sha256": {"design.parquet": sha256(Path("design.parquet"))}}))
        with self.assertRaises(RuntimeError):
            load_frozen(Path("protocol.json"))

    def test_design_refuses_outcome_columns(self):
        synthetic(-3.0)
        pd.read_parquet("design.parquet").assign(day1_chloride=1.0).to_parquet("design.parquet")
        with self.assertRaises(AssertionError):
            design(json.loads(Path("spec.json").read_text()))

    def test_complete_separation_is_not_balance(self):
        treated = np.array([0, 0, 1, 1])
        self.assertEqual(weighted_smd(treated.astype(float), treated, np.ones(4)), float("inf"))
        self.assertEqual(weighted_smd(np.ones(4), treated, np.ones(4)), 0.0)

    def test_final_weights_detect_an_unmeasured_stratum(self):
        rng = np.random.default_rng(1)
        stratum = rng.integers(0, 2, 4000)
        treated = rng.integers(0, 2, 4000)
        measured = ~((treated == 0) & (stratum == 1))                     # controls in stratum 1 are never measured
        covariates = pd.DataFrame({"stratum": stratum.astype(float)})
        result = final_weight_diagnostics(covariates, treated, measured, np.ones(int(measured.sum())))
        self.assertGreater(result["max_abs_smd_against_population"]["control"], 0.5)


    def test_review_needs_this_protocols_completed_estimate(self):
        self.run_case(-3.0)
        protocol, result = Path("protocol.json"), Path("estimate.json")
        forged = {**json.loads(result.read_text()), "protocol_sha256": "different_protocol"}
        result.write_text(json.dumps(forged))
        Path("review.json").write_text(json.dumps({"decision": "approved", "reviewer": "x", "protocol_sha256": sha256(protocol),
                                                   "result_sha256": sha256(result)}))
        with self.assertRaises(RuntimeError):
            review(protocol, Path("review.json"))

    def test_a_renamed_copy_of_the_protocol_shares_its_execution_ledger(self):
        synthetic(-3.0)
        protocol = freeze_and_extract()
        Path("copy.json").write_text(protocol.read_text())
        with self.assertRaises(FileExistsError):
            claim_step(Path("copy.json"), "extract_outcomes")
        with self.assertRaises(FileExistsError):
            record_step(protocol, "extract_outcomes", ["spec.json"], ["outcomes.parquet"])
        with self.assertRaises(RuntimeError):
            record_step(protocol, "unclaimed", ["spec.json"], [])

    def test_structural_observation_nonpositivity_fails_readiness(self):
        x = np.tile([-1.0, 0.0, 1.0], 2000)
        treated = np.repeat([0, 1], 3000)
        measured = (treated == 1) | (x == 0)                       # controls observed only at x = 0
        self.assertEqual(observation_support(pd.DataFrame({"x": x}), treated, measured)["share"], 0.0)

    def test_identical_constants_are_balanced_under_any_weights(self):
        rng = np.random.default_rng(2)
        group = rng.integers(0, 2, 5000)
        self.assertEqual(weighted_smd(np.full(5000, 105.0), group, rng.uniform(0.1, 30, 5000)), 0.0)

    def test_an_analysis_that_cannot_be_computed_is_a_structured_failure(self):
        synthetic(-3.0)
        spec = json.loads(Path("spec.json").read_text())
        frame = pd.read_parquet("design.parquet")
        result = effect(frame, spec, np.full(len(frame), np.nan))
        self.assertIn("failed", result)
        self.assertFalse(diagnostics_pass(result))


    def run_with_outcome_columns(self, **columns) -> dict:
        synthetic(-3.0)
        design_frame = pd.read_parquet("design.parquet").assign(baseline_bicarbonate=24.0)
        design_frame.to_parquet("design.parquet")
        spec = {**json.loads(Path("spec.json").read_text()), "outcomes": ["chloride", "bicarbonate"]}
        Path("spec.json").write_text(json.dumps(spec))
        outcomes = pd.read_parquet("outcomes.parquet").assign(**{"day1_bicarbonate": 25.0, **columns})
        outcomes.to_parquet("outcomes.parquet")
        return estimate(freeze_and_extract())[0]

    def test_an_unavailable_primary_or_negative_control_writes_a_failed_report(self):
        for column in ("day1_chloride", "nc_prior_chloride"):
            with self.subTest(column=column):
                report = self.run_with_outcome_columns(**{column: np.nan})
                self.assertEqual(report["readiness_status"], "failed")
                self.assertTrue(Path("estimate.json").exists())
                for name in ("estimate.json", "outcomes.parquet", "protocol.json"):
                    Path(name).unlink()
                shutil.rmtree(frozen_protocol.LEDGER)                # same protocol digest: a fresh trial needs a fresh ledger

    def test_an_unavailable_descriptive_outcome_is_only_a_secondary_failure(self):
        report = self.run_with_outcome_columns(day1_bicarbonate=np.nan)
        self.assertIn("failed", report["results"]["bicarbonate_change"])
        self.assertEqual(report["readiness_status"], "ready_for_external_review")

if __name__ == "__main__":
    unittest.main()
