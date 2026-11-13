import unittest

from sparse_irregular_registry import build_sparse_registry


def value_audit(*, forward=True):
    model = {"significant": True}
    return {
        "results": [
            {
                "target": "inr",
                "time_bin": "6_24h",
                "patient_splits": [{"log_event": model} for _ in range(7)],
                "care_unit_holdout": {"log_event": model},
                "forward_trajectory_holdout": {
                    "log_event": {"significant": forward}
                },
            }
        ]
    }


class SparseIrregularRegistryTests(unittest.TestCase):
    def test_missing_external_database_evidence_fails_closed(self):
        result = build_sparse_registry(
            value_audit(),
            {
                "calibrated_cells": [
                    {"target": "inr", "time_bin": "6_24h", "model": "log_event"}
                ]
            },
        )
        cell = result["target_horizon_registry"]["inr@12h"]
        self.assertFalse(cell["can_move"])
        self.assertIn("external_database", cell["rejection_reasons"])

    def test_all_source_specific_gates_allow_sparse_event_model(self):
        evidence = {
            "validated_cells": [
                {"target": "inr", "time_bin": "6_24h", "model": "log_event"}
            ]
        }
        result = build_sparse_registry(
            value_audit(),
            {
                "calibrated_cells": [
                    {"target": "inr", "time_bin": "6_24h", "model": "log_event"}
                ]
            },
            external_database_audit=evidence,
        )
        cell = result["target_horizon_registry"]["inr@12h"]
        self.assertTrue(cell["can_move"])
        self.assertEqual(cell["source_model"], "sparse_irregular_log_event")
        self.assertFalse(cell["validated_for_joint_runtime"])


if __name__ == "__main__":
    unittest.main()
