import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from bun_forecast_demo.app import app
from bun_forecast_demo.examples import EXAMPLES, example_path
from bun_forecast_demo.runtime import (
    HORIZON_HOURS,
    capabilities,
    predict_csv,
    prepare_trajectory,
)
from bun_forecast_demo.evaluate_local_patients import (
    _history_indices,
    _patient_equalized,
)


SAMPLE = example_path("progressive")


class BunForecastRuntimeTests(unittest.TestCase):
    def test_sample_calls_exact_validated_three_hour_artifact(self):
        result = predict_csv(SAMPLE.read_bytes())
        self.assertEqual(result["target"], "bun")
        self.assertEqual(result["horizon_hours"], HORIZON_HOURS)
        self.assertEqual(result["source_model"], "patient_state_precision_bun3_v1")
        self.assertLess(result["lower"], result["personalized_point"])
        self.assertGreater(result["upper"], result["personalized_point"])
        self.assertAlmostEqual(
            result["upper"] - result["lower"],
            result["interval_full_width"],
            places=3,
        )
        self.assertFalse(result["clinical_claim_allowed"])
        self.assertFalse(result["causal_claim_allowed"])
        self.assertFalse(result["patient_data_persisted"])

    def test_every_synthetic_example_runs_through_the_real_model(self):
        for example in EXAMPLES:
            with self.subTest(example=example["id"]):
                result = predict_csv(example_path(example["id"]).read_bytes())
                self.assertEqual(result["target"], "bun")
                self.assertTrue(np.isfinite(result["personalized_point"]))
                self.assertLess(result["lower"], result["upper"])

    def test_sparse_example_reports_structured_missingness(self):
        result = predict_csv(example_path("sparse").read_bytes())
        warning_codes = {warning["code"] for warning in result["input_quality"]["warnings"]}
        self.assertIn("missing_treatment_context", warning_codes)
        self.assertGreater(result["current_bun_measurement_age_hours"], 0.0)

    def test_missing_values_are_carried_with_measurement_age_not_zero(self):
        raw = pd.DataFrame({
            "hours_since_onset": [0, 2, 4, 6, 8, 10],
            "bun": [20, 21, 22, 23, 24, None],
            "creatinine": [1.0, None, None, 1.2, None, None],
        })
        frame = prepare_trajectory(raw)
        self.assertEqual(frame["bun_t"].iloc[-1], 24.0)
        self.assertEqual(frame["bun_age_hr"].iloc[-1], 2.0)
        self.assertEqual(frame["creatinine_t"].iloc[-1], 1.2)
        self.assertEqual(frame["creatinine_age_hr"].iloc[-1], 4.0)
        self.assertTrue(frame["hist_fluids"].isna().all())

    def test_short_history_is_rejected(self):
        raw = pd.DataFrame({
            "hours_since_onset": [0, 1, 2],
            "bun": [20, 21, 22],
        })
        with self.assertRaisesRegex(ValueError, "at least 6"):
            prepare_trajectory(raw)

    def test_capability_contract_does_not_relabel_bun6(self):
        contract = capabilities()
        self.assertEqual(contract["horizon_hours"], 3)
        self.assertIn("BUN@6h", contract["reference_note"])
        self.assertIn("BUN@3h", contract["reference_note"])

    def test_evaluation_history_never_crosses_stays(self):
        frame = pd.DataFrame({
            "stay_id": ["a"] * 6 + ["b"] * 7,
            "t": pd.date_range("2026-01-01", periods=13, freq="3h"),
        })
        indices, complete = _history_indices(frame, 6)
        self.assertTrue(complete[5])
        self.assertTrue(complete[11])
        self.assertEqual(indices[5].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(indices[11].tolist(), [6, 7, 8, 9, 10, 11])
        self.assertFalse(complete[6])

    def test_patient_equalized_metric_does_not_overweight_long_stays(self):
        value, by_patient = _patient_equalized(
            np.asarray(["a", "a", "a", "b"]),
            np.asarray([1.0, 1.0, 1.0, 5.0]),
        )
        self.assertEqual(by_patient.tolist(), [1.0, 5.0])
        self.assertEqual(value, 3.0)


class BunForecastApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_csv_upload_endpoint(self):
        response = self.client.post(
            "/api/predict-csv",
            content=SAMPLE.read_bytes(),
            headers={"Content-Type": "text/csv; charset=utf-8"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["horizon_hours"], 3)

    def test_invalid_csv_returns_400(self):
        response = self.client.post(
            "/api/predict-csv",
            content=b"hours_since_onset,bun\n0,20\n",
            headers={"Content-Type": "text/csv"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("at least 6", response.json()["detail"])

    def test_example_catalog_and_downloads(self):
        response = self.client.get("/api/examples")
        self.assertEqual(response.status_code, 200)
        examples = response.json()["examples"]
        self.assertEqual(len(examples), 5)
        self.assertTrue(all(example["synthetic"] for example in examples))
        self.assertTrue(all("zh-Hant" in example["title"] for example in examples))
        self.assertTrue(all("en" in example["title"] for example in examples))
        self.assertEqual(
            [example["title"]["zh-Hant"] for example in examples],
            [f"病人 {index}" for index in range(1, 6)],
        )
        self.assertTrue(all("description" not in example for example in examples))
        for example in examples:
            with self.subTest(example=example["id"]):
                download = self.client.get(example["download_url"])
                self.assertEqual(download.status_code, 200)
                self.assertIn("hours_since_onset,bun", download.text)

    def test_unknown_example_and_removed_legacy_sample_fail_closed(self):
        self.assertEqual(self.client.get("/examples/not-a-real-example.csv").status_code, 404)
        self.assertEqual(self.client.get("/sample.csv").status_code, 404)

    def test_home_page_exposes_upload_ui(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("預測 3 小時後的 BUN", response.text)
        self.assertIn("Forecast BUN three hours ahead", response.text)
        self.assertIn("中文", response.text)
        self.assertIn("English", response.text)
        self.assertIn('id="example-select"', response.text)
        self.assertIn('type="file"', response.text)


if __name__ == "__main__":
    unittest.main()
