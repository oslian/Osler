import unittest

from eicu_to_mimic_cell_transfer_audit import build_report


class TransferAuditTest(unittest.TestCase):
    def test_separates_point_and_interval_transfer(self):
        eicu = {
            "modules": [
                {
                    "module": "renal",
                    "validated": {
                        "nowcast_targets": [],
                        "forecast_targets": ["potassium", "creatinine"],
                        "interval_targets": ["potassium"],
                    },
                }
            ]
        }
        mimic = {
            "validated": {
                "nowcast_targets": [],
                "forecast_targets": ["potassium"],
                "interval_targets": ["potassium"],
            },
            "targets": {
                "creatinine": {
                    "forecast": {
                        "random_patient_splits": {
                            "evaluated_splits": 7,
                            "selected_ridge_splits": 0,
                            "heldout_beats_baseline_splits": 0,
                            "heldout_beats_placebo_splits": 0,
                            "interval_pass_splits": 0,
                        },
                        "careunit_holdout": {
                            "available": True,
                            "gate_passed": False,
                            "interval_gate_passed": False,
                        },
                        "time_holdout": {
                            "available": True,
                            "gate_passed": False,
                            "interval_gate_passed": False,
                        },
                    }
                }
            },
        }
        catalog = {
            "candidates": [
                {"validated_cells": ["potassium@6h", "creatinine@6h"]}
            ]
        }

        report = build_report(eicu, mimic, catalog)

        self.assertEqual(
            report["current_runtime_6h"]["point_forecast_pass_cells"],
            ["potassium@6h"],
        )
        self.assertEqual(
            report["current_runtime_6h"]["point_and_interval_pass_cells"],
            ["potassium@6h"],
        )
        self.assertEqual(
            report["current_runtime_6h"]["point_forecast_fail_cells"][0][
                "cell"
            ],
            "creatinine@6h",
        )
        self.assertFalse(report["frozen_eicu_weight_transfer_tested"])


if __name__ == "__main__":
    unittest.main()
