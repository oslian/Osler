import unittest

from scripts.build_multisource_shared_state_registry import build_registry


class MultisourceSharedStateRegistryTest(unittest.TestCase):
    def test_distinguishes_state_consumption_from_incremental_latent_value(self):
        audit = {
            "development_databases": ["eICU", "MIMIC-IV"],
            "external_lockbox": "NWICU",
            "cells": [
                {
                    "cell": "map@1h",
                    "status": "evaluated",
                    "selection": {"feature_view": "all"},
                    "external": {"point_transfer_pass": True},
                    "external_pass": False,
                },
                {
                    "cell": "platelets@3h",
                    "status": "evaluated",
                    "selection": {"feature_view": "shared"},
                    "external": {"point_transfer_pass": True},
                    "external_pass": True,
                },
                {
                    "cell": "urine_output@1h",
                    "status": "not_evaluable_on_nwicu",
                    "reason": "target_unavailable",
                    "external_pass": False,
                },
            ],
        }
        intervals = {"cells": {
            "map@1h": {"corrected_nwicu_point_gate": {"pass": True}, "runtime_interval_eligible": True, "failure_reasons": []},
            "platelets@3h": {"corrected_nwicu_point_gate": {"pass": False}, "runtime_interval_eligible": False, "failure_reasons": ["nwicu_point"]},
        }}
        result = build_registry(audit, intervals)
        self.assertEqual(result["shared_state_consuming_cells"], 2)
        self.assertEqual(result["shared_state_point_cells"], 1)
        self.assertEqual(result["calibrated_interval_cells"], 1)
        self.assertEqual(result["latent_incremental_cells"], 0)
        self.assertTrue(result["latent_incremental_revalidation_required"])
        self.assertFalse(result["requested_point_coverage_goal_met"])
        self.assertEqual(result["not_externally_evaluable_cells"], 1)
        self.assertFalse(result["runtime_replacement_allowed"])


if __name__ == "__main__":
    unittest.main()
