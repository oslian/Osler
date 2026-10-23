import unittest

from compile_mimic_crossdb_rescue import build_report


def interval(pass_gate=True):
    return {"coverage_gate_passed": pass_gate}


class CompileMimicCrossdbRescueTests(unittest.TestCase):
    def test_compiler_combines_point_and_interval_rescues(self):
        base = {"current_runtime_6h": {
            "eicu_validated_cells": ["glucose@6h", "heart_rate@6h", "creatinine@6h"],
            "point_forecast_pass_cells": ["heart_rate@6h"],
            "point_and_interval_pass_cells": [],
        }}
        rescue = {"targets": {
            "glucose": {
                "validated": True,
                "random_patient_conformal_90": {"candidate_coverage_pass_splits": 7},
                "careunit_holdout": {"conformal_90": {"candidate": interval()}},
                "time_holdout": {"conformal_90": {"candidate": interval()}},
            },
            "heart_rate": {
                "validated": False,
                "random_patient_conformal_90": {"baseline_coverage_pass_splits": 7},
                "careunit_holdout": {"conformal_90": {"baseline": interval()}},
                "time_holdout": {"conformal_90": {"baseline": interval()}},
            },
        }}
        failed = {"validated": {"forecast_targets": []}}

        report = build_report(base, rescue, failed, failed)

        self.assertEqual(report["counts"]["mimic_point_pass"], 2)
        self.assertEqual(report["counts"]["mimic_point_and_interval_pass"], 2)
        self.assertEqual(report["fallback_cells"], ["creatinine@6h"])


if __name__ == "__main__":
    unittest.main()
