from pathlib import Path
import unittest

from scripts.audit_release_tolerance_intervals import ACCEPTED_COVERAGE, audit_release
from whole_body_jepa import predict_patient


ROOT = Path(__file__).resolve().parents[1]


class ReleaseToleranceIntervalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = audit_release(ROOT / "models")

    def test_runtime_interval_count_matches_manifest(self) -> None:
        summary = self.audit["summary"]
        self.assertGreater(summary["resolved_runtime_interval_cells"], 0)
        self.assertTrue(summary["count_matches_manifest"])

    def test_all_runtime_intervals_pass_empirical_coverage_gates(self) -> None:
        failed = [
            item["cell"]
            for item in self.audit["cells"]
            if not item["empirical_coverage_gate_pass"]
        ]
        self.assertEqual(failed, [])
        self.assertTrue(self.audit["summary"]["coverage_calibration_release_ready"])

    def test_reported_coverage_is_empirical_and_bounded(self) -> None:
        low, high = ACCEPTED_COVERAGE
        for item in self.audit["cells"]:
            values = [
                value
                for key, value in item.items()
                if key.endswith("coverage") and key != "nominal_coverage" and value is not None
            ]
            self.assertTrue(values, item["cell"])
            for value in values:
                self.assertGreaterEqual(value, low, item["cell"])
                self.assertLessEqual(value, high, item["cell"])

    def test_runtime_never_emits_malformed_interval(self) -> None:
        result = predict_patient(
            (ROOT / "examples/patient_2.csv").read_text(),
            filename="patient_2.csv",
            model_dir=ROOT / "models",
        )
        for cell in result["cells"]:
            if cell["lower"] is None or cell["upper"] is None:
                self.assertIsNone(cell["lower"])
                self.assertIsNone(cell["upper"])
                continue
            self.assertIsNotNone(cell["forecast_point"])
            self.assertLessEqual(cell["lower"], cell["forecast_point"])
            self.assertGreaterEqual(cell["upper"], cell["forecast_point"])
            self.assertFalse(cell["abstained"])

    def test_distributional_readiness_gaps_remain_explicit(self) -> None:
        summary = self.audit["summary"]
        self.assertEqual(summary["crps_vs_persistence_available_cells"], 0)
        self.assertEqual(summary["applicability_range_available_cells"], 0)
        self.assertFalse(summary["runtime_api_exposes_empirical_coverage"])
        self.assertFalse(summary["distributional_release_complete"])

    def test_audit_is_factual_and_nonclinical(self) -> None:
        self.assertFalse(self.audit["causal_claim_allowed"])
        self.assertFalse(self.audit["clinical_promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
