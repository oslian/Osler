"""WholeBody runtime adapter contract: 41-cell registry, statuses, null intervals, gating."""

import copy
import json
from pathlib import Path
import unittest

from oslian_demo.monitoring import anatomy
from oslian_demo.monitoring.timeline import ModelInput
from oslian_demo.monitoring.wholebody import (
    RuntimeConfig, RuntimeUnavailable, WholeBodyRuntime, head_abstained, load_contract_fixture,
    normalize_release_cells,
)

FIXTURE_CONFIG = RuntimeConfig("fixture", None, "python", None)


class ContractFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract_fixture()
        cls.catalog = {c["cell"]: c for c in cls.contract["catalog"]["cells"]}
        cls.sample_cells = cls.contract["sample"]["release_output"]["cells"]

    def test_fixture_provenance_is_pinned_and_verified(self):
        self.assertEqual(self.contract["model_commit"], "5a49b3969d69145efa8ef66d0c6e80d24daef5d2")
        self.assertEqual(self.contract["registry_sha256"], self.contract["manifest"]["files"]["runtime_registry.json"]["sha256"])
        self.assertEqual(self.contract["scikit_learn_pin"], "1.7.2")
        self.assertTrue(all(c["status"] == "passed" for c in self.contract["worker_checks"]))
        self.assertFalse(self.contract["formal_architecture_gate"]["release_satisfies_rule"])
        self.assertFalse(self.contract["catalog"]["causal_claim_allowed"])

    def test_catalog_counts_come_from_the_registry_not_constants(self):
        cells = self.contract["catalog"]["cells"]
        self.assertEqual(len(cells), self.contract["catalog"]["catalog_cells"])
        self.assertEqual(sum(1 for c in cells if c["can_move"]), self.contract["catalog"]["externally_validated_point_cells"])
        self.assertEqual(sum(1 for c in cells if not c["can_move"]), self.contract["catalog"]["fallback_cells"])
        self.assertEqual(len(cells), len(self.contract["manifest"]["files"]) and int(self.contract["manifest"]["catalog_cells"]))
        for cell in cells:
            self.assertTrue(cell["unit"], cell["cell"])
            if not cell["can_move"]:
                self.assertEqual(cell["fallback_reason"], "shared_jepa_head_not_external_validated")

    def test_normalized_cells_follow_the_contract(self):
        state = {"glucose": {"count": 5}}
        cells = normalize_release_cells(self.sample_cells, state, self.catalog)
        self.assertEqual(len(cells), len(self.catalog))
        # can_move is the catalog's capability bit; a head that declined to move still
        # carries it, so the expected count comes from the sample's own rows.
        expected_model = sum(
            1 for c in self.sample_cells
            if c.get("can_move")
            and not head_abstained(str(c.get("source_model") or ""), c.get("forecast_point"), c.get("observed_value"))
        )
        self.assertEqual(sum(1 for c in cells if c["status"] == "model"), expected_model)
        self.assertEqual(sum(1 for c in cells if c["status"] == "persistence"), len(self.catalog) - expected_model)
        for cell in cells:
            self.assertIn(cell["status"], {"model", "persistence", "missing", "degraded"})
            self.assertIsNone(cell["lower"])
            self.assertIsNone(cell["upper"])
            self.assertFalse(cell["causal_claim_allowed"])
            self.assertEqual(cell["interval_status"], "not_bundled_for_this_runtime_cell")
            if cell["status"] == "persistence":
                self.assertAlmostEqual(cell["forecast_point"], cell["observed_value"])
                self.assertTrue(cell["fallback_reason"])
                if cell["fallback_reason"] == "shared_jepa_head_abstained":
                    self.assertEqual(cell["source_model"], "unified_shared_jepa_target_head")
                else:
                    self.assertEqual(cell["source_model"], "patient_persistence_anchor")
            if cell["status"] == "model":
                self.assertIn(cell["source_model"], {"frozen_whole_body_jepa", "unified_shared_jepa_target_head"})
                self.assertIsNone(cell["fallback_reason"])
        glucose = next(c for c in cells if c["cell"] == "glucose@1h")
        self.assertEqual(glucose["ledger_observation_count"], 5)

    def test_a_shared_head_that_declined_to_move_is_not_badged_as_a_model_prediction(self):
        raw = copy.deepcopy(self.sample_cells)
        target = next(c for c in raw if c["cell"] == "chloride@12h")
        self.assertEqual(target["source_model"], "unified_shared_jepa_target_head")
        # The release zeroes a sub-threshold delta and returns current + 0.
        target.update({"forecast_point": target["observed_value"], "can_move": True, "fallback_reason": None})
        cells = normalize_release_cells(raw, {}, self.catalog)
        chloride = next(c for c in cells if c["cell"] == "chloride@12h")
        self.assertEqual(chloride["status"], "persistence")
        self.assertEqual(chloride["fallback_reason"], "shared_jepa_head_abstained")
        # The head that declined stays on the record; only the badge changes.
        self.assertEqual(chloride["source_model"], "unified_shared_jepa_target_head")

    def test_a_frozen_backbone_forecast_equal_to_the_observation_stays_a_model_prediction(self):
        # hypoxemia_event is binary: predicting 0.0 on an observed 0.0 is a real
        # forecast, and the frozen path carries no abstention threshold.
        event = next(c for c in self.sample_cells if c["cell"] == "hypoxemia_event@1h")
        self.assertEqual(event["source_model"], "frozen_whole_body_jepa")
        self.assertEqual(event["forecast_point"], event["observed_value"])
        cells = normalize_release_cells(copy.deepcopy(self.sample_cells), {}, self.catalog)
        got = next(c for c in cells if c["cell"] == "hypoxemia_event@1h")
        self.assertEqual(got["status"], "model")

    def test_missing_target_history_is_missing_not_invented(self):
        raw = copy.deepcopy(self.sample_cells)
        target = next(c for c in raw if c["cell"] == "platelets@1h")
        target.update({"observed_value": None, "forecast_point": None, "source_model": "missing",
                       "can_move": False, "fallback_reason": "no_observed_target_value", "observations": 0})
        cells = normalize_release_cells(raw, {}, self.catalog)
        platelets = next(c for c in cells if c["cell"] == "platelets@1h")
        self.assertEqual(platelets["status"], "missing")
        self.assertIsNone(platelets["forecast_point"])

    def test_intervals_and_foreign_cells_fail_closed(self):
        raw = copy.deepcopy(self.sample_cells)
        raw[0]["lower"], raw[0]["upper"] = 1.0, 2.0
        with self.assertRaises(RuntimeUnavailable):
            normalize_release_cells(raw, {}, self.catalog)
        raw = copy.deepcopy(self.sample_cells)
        raw[0]["cell"] = "sodium@1h"
        with self.assertRaises(RuntimeUnavailable):
            normalize_release_cells(raw, {}, self.catalog)
        with self.assertRaises(RuntimeUnavailable):
            normalize_release_cells(self.sample_cells[:-1], {}, self.catalog)


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
        hypox = next(c for c in result.cells if c["cell"] == "hypoxemia_event@1h")
        self.assertEqual(hypox["observed_value"], 1.0)
        self.assertEqual(result.summary["degraded_cells"], len(self.runtime.catalog))

    def test_capabilities_are_registry_driven(self):
        cap = self.runtime.capabilities()
        self.assertEqual(cap["catalog_cells"], len(cap["cells"]))
        self.assertEqual(cap["model_cells"] + cap["persistence_cells"], cap["catalog_cells"])
        self.assertEqual(len(cap["variables"]), 53)
        self.assertFalse(cap["causal_claim_allowed"])
        self.assertFalse(cap["clinical_alerts_supported"])
        self.assertFalse(cap["runtime_attached"])
        self.assertIn("not anatomical organs", cap["modules_note"])
        # 2026-09-01 decision: bounded research-demo runtime only; nothing about the formal rule changes
        self.assertEqual(cap["runtime_scope"], "gated fixture (safe default)")
        approval = cap["approval"]
        self.assertEqual(approval["approved_scope"], "bounded research-demo runtime")
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
