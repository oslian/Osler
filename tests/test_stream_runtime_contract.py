"""Release runtime adapter contract: 41-cell catalog, statuses, calibrated intervals, events, gating."""

import copy
import json
from pathlib import Path
import unittest

from oslian_demo.config import load_model_lock
from oslian_demo.monitoring import anatomy
from oslian_demo.monitoring.timeline import ModelInput
from oslian_demo.monitoring.wholebody import (
    ABSTENTION_REASON, RuntimeConfig, RuntimeUnavailable, WholeBodyRuntime, load_contract_fixture,
    normalize_release_cells,
)

FIXTURE_CONFIG = RuntimeConfig("fixture", None, "python", None)
MODEL_SOURCES = {"frozen_whole_body_jepa", "unified_shared_jepa_target_head",
                 "multisource_shared_jepa_target_head", "calibrated_event_risk_head"}


class ContractFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract_fixture()
        cls.lock = load_model_lock()
        cls.catalog = {c["cell"]: c for c in cls.contract["catalog"]["cells"]}
        cls.sample_cells = cls.contract["sample"]["release_output"]["cells"]
        cls.bounded = frozenset(cls.lock["output_boundaries"]["interval_statuses_with_bounds"])

    def normalize(self, raw, state=None):
        return normalize_release_cells(raw, state or {}, self.catalog, self.bounded)

    def test_fixture_provenance_is_pinned_and_verified(self):
        self.assertEqual(self.contract["model_commit"], self.lock["validated_commit"])
        self.assertEqual(self.contract["model_repository"], self.lock["repository"])
        files = self.contract["manifest"]["files"]
        self.assertEqual(self.contract["registry_sha256"], files["runtime_registry.json"]["sha256"])
        self.assertEqual(self.contract["selector_sha256"], files["unified_source_selector.json"]["sha256"])
        self.assertEqual(self.contract["scikit_learn_pin"], "1.7.2")
        self.assertTrue(all(c["status"] == "passed" for c in self.contract["worker_checks"]))
        self.assertFalse(self.contract["formal_architecture_gate"]["release_satisfies_rule"])
        self.assertFalse(self.contract["catalog"]["causal_claim_allowed"])

    def test_catalog_counts_agree_with_the_release_manifest(self):
        cells = self.contract["catalog"]["cells"]
        manifest = self.contract["manifest"]
        self.assertEqual(len(cells), int(manifest["catalog_cells"]))
        self.assertEqual(sum(c["can_move"] for c in cells), int(manifest["runtime_move_cells"]))
        self.assertEqual(sum(c["interval_capable"] for c in cells), int(manifest["runtime_interval_cells"]))
        self.assertEqual(sum(not c["can_move"] for c in cells), int(manifest["fallback_cells"]))
        for cell in cells:
            self.assertTrue(cell["unit"], cell["cell"])
            if cell["interval_capable"]:
                self.assertTrue(cell["can_move"], cell["cell"])
            if cell["can_move"]:
                self.assertIsNone(cell["fallback_reason"], cell["cell"])
            else:
                self.assertTrue(cell["fallback_reason"], cell["cell"])
            if cell["target"] == "hypoxemia_event":
                self.assertEqual(cell["unit"], "probability")
                self.assertFalse(cell["interval_capable"])

    def test_normalized_cells_follow_the_contract(self):
        cells = self.normalize(self.sample_cells, {"glucose": {"count": 5}})
        self.assertEqual(len(cells), len(self.catalog))
        expected_model = sum(1 for c in self.sample_cells
                             if c.get("can_move") and c.get("forecast_point") is not None
                             and c["source_model"] != "patient_persistence_anchor")
        self.assertEqual(sum(c["status"] == "model" for c in cells), expected_model)
        for cell in cells:
            self.assertIn(cell["status"], {"model", "persistence", "missing", "degraded"})
            self.assertFalse(cell["causal_claim_allowed"])
            spec = self.catalog[cell["cell"]]
            if cell["lower"] is not None:
                self.assertTrue(spec["interval_capable"], cell["cell"])
                self.assertEqual(cell["status"], "model")
                self.assertIn(cell["interval_status"], self.bounded)
                self.assertLessEqual(cell["lower"], cell["forecast_point"])
                self.assertLessEqual(cell["forecast_point"], cell["upper"])
            else:
                self.assertIsNone(cell["upper"])
                self.assertNotIn(cell["interval_status"], self.bounded)
            if cell["status"] == "persistence":
                self.assertAlmostEqual(cell["forecast_point"], cell["observed_value"])
                self.assertEqual(cell["source_model"], "patient_persistence_anchor")
                self.assertTrue(cell["fallback_reason"])
            if cell["status"] == "model":
                self.assertIn(cell["source_model"], MODEL_SOURCES)
                self.assertIsNone(cell["fallback_reason"])
        glucose = next(c for c in cells if c["cell"] == "glucose@1h")
        self.assertEqual(glucose["ledger_observation_count"], 5)
        # the synthetic patient exercises every calibrated cell
        self.assertEqual(sum(c["lower"] is not None for c in cells), self.contract["catalog"]["interval_cells"])

    def test_event_cells_are_probabilities_without_intervals(self):
        cells = self.normalize(copy.deepcopy(self.sample_cells))
        events = [c for c in cells if c["target"] == "hypoxemia_event"]
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertEqual(event["status"], "model")
            self.assertEqual(event["source_model"], "calibrated_event_risk_head")
            self.assertGreaterEqual(event["forecast_point"], 0.0)
            self.assertLessEqual(event["forecast_point"], 1.0)
            self.assertIsNone(event["lower"])
            self.assertIsNone(event["upper"])

    def test_a_head_the_release_reports_abstained_is_a_baseline(self):
        raw = copy.deepcopy(self.sample_cells)
        target = next(c for c in raw if c["cell"] == "chloride@12h")
        target.update({"forecast_point": target["observed_value"], "can_move": False, "lower": None, "upper": None,
                       "source_model": "patient_persistence_anchor", "fallback_reason": ABSTENTION_REASON,
                       "interval_status": "interval_external_validation_not_passed"})
        chloride = next(c for c in self.normalize(raw) if c["cell"] == "chloride@12h")
        self.assertEqual(chloride["status"], "persistence")
        self.assertEqual(chloride["fallback_reason"], "continuous_head_abstained")

    def test_missing_target_history_is_missing_not_invented(self):
        raw = copy.deepcopy(self.sample_cells)
        target = next(c for c in raw if c["cell"] == "platelets@1h")
        target.update({"observed_value": None, "forecast_point": None, "source_model": "missing", "lower": None,
                       "upper": None, "can_move": False, "fallback_reason": "no_observed_target_value",
                       "interval_status": "missing_point", "observations": 0})
        platelets = next(c for c in self.normalize(raw) if c["cell"] == "platelets@1h")
        self.assertEqual(platelets["status"], "missing")
        self.assertIsNone(platelets["forecast_point"])

    def _bounded_raw(self):
        raw = copy.deepcopy(self.sample_cells)
        return raw, next(c for c in raw if c["lower"] is not None)

    def test_a_bound_the_catalog_does_not_allow_fails_closed(self):
        raw = copy.deepcopy(self.sample_cells)
        unbounded = next(c for c in raw if c["can_move"] and not self.catalog[c["cell"]]["interval_capable"]
                         and c["target"] != "hypoxemia_event")
        unbounded.update({"lower": unbounded["forecast_point"] - 1, "upper": unbounded["forecast_point"] + 1,
                          "interval_status": "externally_audited_mimic_split_conformal_90"})
        with self.assertRaises(RuntimeUnavailable):
            self.normalize(raw)

    def test_a_bound_on_a_baseline_fails_closed(self):
        raw = copy.deepcopy(self.sample_cells)
        baseline = next(c for c in raw if c["source_model"] == "patient_persistence_anchor")
        baseline.update({"lower": 0.0, "upper": 1e9, "interval_status": "externally_audited_mimic_split_conformal_90"})
        with self.assertRaises(RuntimeUnavailable):
            self.normalize(raw)

    def test_a_bound_that_misses_its_point_or_is_not_finite_fails_closed(self):
        raw, cell = self._bounded_raw()
        cell["lower"], cell["upper"] = cell["forecast_point"] + 1, cell["forecast_point"] + 2
        with self.assertRaises(RuntimeUnavailable):
            self.normalize(raw)
        raw, cell = self._bounded_raw()
        cell["upper"] = float("inf")
        with self.assertRaises(RuntimeUnavailable):
            self.normalize(raw)

    def test_a_calibrated_status_without_bounds_fails_closed(self):
        raw, cell = self._bounded_raw()
        cell["lower"] = cell["upper"] = None
        with self.assertRaises(RuntimeUnavailable):
            self.normalize(raw)

    def test_unknown_sources_bad_probabilities_and_foreign_cells_fail_closed(self):
        raw = copy.deepcopy(self.sample_cells)
        next(c for c in raw if c["source_model"] == "frozen_whole_body_jepa")["source_model"] = "some_new_head"
        with self.assertRaises(RuntimeUnavailable):
            self.normalize(raw)
        raw = copy.deepcopy(self.sample_cells)
        next(c for c in raw if c["target"] == "hypoxemia_event")["forecast_point"] = 1.5
        with self.assertRaises(RuntimeUnavailable):
            self.normalize(raw)
        raw = copy.deepcopy(self.sample_cells)
        raw[0]["cell"] = "sodium@1h"
        with self.assertRaises(RuntimeUnavailable):
            self.normalize(raw)
        with self.assertRaises(RuntimeUnavailable):
            self.normalize(self.sample_cells[:-1])


class FixtureModeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = WholeBodyRuntime(FIXTURE_CONFIG)

    def test_default_mode_is_detached_and_degraded(self):
        self.assertEqual(self.runtime.mode, "fixture")
        self.assertFalse(self.runtime.attached())
        health = self.runtime.health()
        self.assertFalse(health["ready"])
        self.assertEqual(health["status"], "runtime_not_attached")
        model_input = ModelInput("2026-03-14T08:00:00.000Z", "2026-03-14T08:00:00.000Z", "hours_since_onset,variable,value\n",
                                 0, (), "hash", "2026-03-14T06:00:00.000Z")
        state = {"o2sat": {"value": 91.0, "age_hours": 0.0, "count": 2}, "map": {"value": 70.0, "age_hours": 1.0, "count": 3}}
        result = self.runtime.run(model_input, state)
        self.assertEqual(len(result.cells), len(self.runtime.catalog))
        self.assertTrue(all(c["status"] == "degraded" and c["forecast_point"] is None for c in result.cells))
        self.assertTrue(all(c["fallback_reason"] == "wholebody_runtime_not_attached" for c in result.cells))
        self.assertTrue(all(c["lower"] is None and c["upper"] is None and c["interval_status"] == "runtime_not_attached"
                            for c in result.cells))
        hypox = next(c for c in result.cells if c["cell"] == "hypoxemia_event@1h")
        self.assertEqual(hypox["observed_value"], 1.0)
        self.assertEqual(result.summary["degraded_cells"], len(self.runtime.catalog))

    def test_capabilities_are_registry_driven(self):
        cap = self.runtime.capabilities()
        self.assertEqual(cap["catalog_cells"], len(cap["cells"]))
        self.assertEqual(cap["model_cells"] + cap["persistence_cells"], cap["catalog_cells"])
        lock = load_model_lock()["runtime_contract"]
        self.assertEqual(cap["model_cells"], lock["model_capable_cells"])
        self.assertEqual(cap["interval_cells"], lock["interval_cells"])
        self.assertEqual(cap["event_risk_cells"], lock["event_risk_cells"])
        self.assertIn("own calibrator", cap["interval_policy"])
        self.assertEqual(len(cap["variables"]), 53)
        self.assertFalse(cap["causal_claim_allowed"])
        self.assertFalse(cap["clinical_alerts_supported"])
        self.assertFalse(cap["runtime_attached"])
        self.assertIn("not anatomical organs", cap["modules_note"])
        # 2026-09-01 decision, 2026-11-12 release adoption: bounded research-demo runtime only;
        # nothing about the formal rule changes
        self.assertEqual(cap["runtime_scope"], "gated fixture (safe default)")
        approval = cap["approval"]
        self.assertEqual(approval["approved_scope"], "bounded research-demo runtime")
        self.assertEqual(approval["release_adopted_on"], "2026-11-12")
        self.assertFalse(approval["formal_architecture_rule_satisfied"])
        self.assertFalse(approval["promotion_registry_entry"])
        self.assertTrue(approval["fail_closed_checks_unchanged"])
        self.assertEqual(approval["safe_default_mode"], "fixture")
        for phrase in ("formal adjacent-three-level model", "production runtime", "clinically validated model"):
            self.assertIn(phrase, approval["must_not_be_called"])

    def test_env_config_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            RuntimeConfig.from_env({"OSLIAN_WHOLEBODY_RUNTIME": "magic"})
        self.assertEqual(RuntimeConfig.from_env({}).mode, "fixture")


class AnatomyTests(unittest.TestCase):
    def test_organ_map_is_anatomical_and_cells_join_the_catalog(self):
        contract = load_contract_fixture()
        document = anatomy.anatomy_document(contract["catalog"]["cells"])
        ids = [o["id"] for o in document["organs"]]
        self.assertEqual(set(ids), set(anatomy.CANONICAL_ORGANS))
        for organ in document["organs"]:
            for word in anatomy.FUNCTIONAL_SYSTEM_WORDS:
                self.assertNotIn(word, organ["id"])
        kidneys = next(o for o in document["organs"] if o["id"] == "kidneys")
        self.assertIn("creatinine@6h", {c["cell"] for c in kidneys["cells"]})
        lungs = next(o for o in document["organs"] if o["id"] == "lungs")
        self.assertIn("hypoxemia_event@1h", {c["cell"] for c in lungs["cells"]})
        brain = next(o for o in document["organs"] if o["id"] == "brain")
        self.assertEqual(brain["cells"], [])
        self.assertEqual(document["unattributed_variables"], [])
        total = sum(len(o["cells"]) for o in document["organs"]) + len(document["systemic"]["cells"])
        self.assertGreaterEqual(total, len(contract["catalog"]["cells"]), "every cell reaches at least one organ or systemic")

    def test_functional_system_names_are_rejected_as_organs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"schema": "oslian.organ-variable-map.v1", "organs": [
                {"id": "acid_base_electrolyte", "label": "x", "variables": []}], "systemic": {"label": "s", "variables": []}}),
                encoding="utf-8")
            with self.assertRaises(ValueError):
                anatomy.load_organ_map(path)


if __name__ == "__main__":
    unittest.main()
