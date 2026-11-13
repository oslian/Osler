import unittest

from compile_mimic_all_runtime_cells_audit import compile_report, treatment_interval_pass


class CompileMimicAllRuntimeCellsAuditTests(unittest.TestCase):
    def test_treatment_interval_requires_all_available_domains(self):
        report = {
            "validated": True,
            "random_patient_conformal_90": {
                "evaluated_splits": 7,
                "candidate_coverage_pass_splits": 7,
            },
            "careunit_holdout": {"conformal_90": {"candidate": {"coverage_gate_passed": True}}},
            "time_holdout": {"conformal_90": {"candidate": {"coverage_gate_passed": True}}},
        }
        self.assertTrue(treatment_interval_pass(report))
        report["time_holdout"]["conformal_90"]["candidate"]["coverage_gate_passed"] = False
        self.assertFalse(treatment_interval_pass(report))

    def test_validated_hurdle_replaces_failed_continuous_cell(self):
        catalog = {
            "candidates": [{
                "target_horizon_registry": {
                    "glucose@1h": {"task_type": "continuous_value"}
                }
            }]
        }
        hurdle = {
            "reports": {
                "glucose@1h": {
                    "rows": 100,
                    "subjects": 50,
                    "summary": {"validated": True},
                }
            }
        }
        report = compile_report(catalog, {}, {}, {"reports": {}}, [hurdle])
        cell = report["cells"][0]
        self.assertTrue(cell["point_validated"])
        self.assertTrue(cell["hurdle_validated"])
        self.assertEqual(cell["source_model"], "target_specific_change_hurdle")
        self.assertEqual(cell["interval_status"], "validated_90pct")

    def test_complete_physiology_source_beats_incomplete_treatment_source(self):
        catalog = {
            "candidates": [{
                "target_horizon_registry": {
                    "glucose@24h": {"task_type": "continuous_value"}
                }
            }]
        }
        physiology = {
            24: {
                "targets": {
                    "glucose": {
                        "forecast": {
                            "validated": True,
                            "interval_validated": True,
                        }
                    }
                }
            }
        }
        treatment = {
            24: {
                "targets": {
                    "glucose": {
                        "validated": True,
                        "random_patient_conformal_90": {
                            "evaluated_splits": 7,
                            "candidate_coverage_pass_splits": 7,
                        },
                        "careunit_holdout": {
                            "conformal_90": {
                                "candidate": {"coverage_gate_passed": False}
                            }
                        },
                        "time_holdout": {
                            "conformal_90": {
                                "candidate": {"coverage_gate_passed": True}
                            }
                        },
                    }
                }
            }
        }
        report = compile_report(catalog, physiology, treatment, {"reports": {}}, [])
        cell = report["cells"][0]
        self.assertEqual(cell["source_model"], "measurement_pure_ridge")
        self.assertEqual(cell["interval_status"], "validated_90pct")


if __name__ == "__main__":
    unittest.main()
