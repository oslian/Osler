from pathlib import Path
import json
import unittest

import numpy as np

from osler_jepa.shared_release_features import (
    FEATURE_WIDTH,
    build_target_head_features,
    normalize_measurement_state,
)
from whole_body_jepa import load_release, parse_patient_csv, predict_patient


ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def test_physical_zero_is_observed(self) -> None:
        frame = parse_patient_csv(
            "hours_since_onset,urine_output,heart_rate\n0,20,80\n3,0,82\n"
        )
        self.assertEqual(float(frame.iloc[-1]["urine_output"]), 0.0)

    def test_missing_uses_median_but_zero_does_not(self) -> None:
        assets = load_release(ROOT / "models")
        variables = tuple(assets.checkpoint["variables"])
        values = {"urine_output": 0.0, "glucose": None}
        state = normalize_measurement_state(
            values,
            variables=variables,
            medians=assets.scaler.medians,
            scales=assets.scaler.scales,
        )
        urine = variables.index("urine_output")
        glucose = variables.index("glucose")
        self.assertTrue(bool(state.observed[urine]))
        self.assertFalse(bool(state.observed[glucose]))
        self.assertEqual(float(state.normalized[glucose]), 0.0)

    def test_feature_contract_is_311(self) -> None:
        vector = build_target_head_features(
            shared_context=np.zeros(64),
            patient_belief=np.zeros(96),
            target_future=np.zeros(64),
            target_organ=np.zeros(32),
            base_forecast_normalized=1.0,
            measurement_pure_state=np.zeros(53),
            target_current_normalized=2.0,
        )
        self.assertEqual(vector.shape, (FEATURE_WIDTH,))
        self.assertEqual(float(vector[256]), 1.0)
        self.assertEqual(float(vector[-1]), 2.0)

    def test_end_to_end_has_41_cells(self) -> None:
        result = predict_patient(
            (ROOT / "examples/patient_2.csv").read_text(),
            filename="patient_2.csv",
            model_dir=ROOT / "models",
        )
        self.assertEqual(result["summary"]["catalog_cells"], 41)
        event_cells = [
            cell for cell in result["cells"] if cell["target"] == "hypoxemia_event"
        ]
        self.assertEqual(len(event_cells), 2)
        for event in event_cells:
            self.assertTrue(event["can_move"])
            self.assertFalse(event["abstained"])
            self.assertEqual(event["source_model"], "calibrated_event_risk_head")
            self.assertEqual(event["risk_status"], "calibrated_event_probability")
            self.assertEqual(event["event_definition"], "future_o2sat <= 92")
            self.assertEqual(
                event["interval_status"], "not_applicable_calibrated_probability"
            )
            self.assertGreaterEqual(event["event_risk_probability"], 0.0)
            self.assertLessEqual(event["event_risk_probability"], 1.0)
            self.assertEqual(event["forecast_point"], event["event_risk_probability"])
            self.assertIsNone(event["lower"])
            self.assertIsNone(event["upper"])
        calibrated = [
            cell
            for cell in result["cells"]
            if cell["interval_status"] in {
                "externally_audited_mimic_split_conformal_90", "unified_source_disjoint_calibration_90"
            }
        ]
        assets = load_release(ROOT / "models")
        expected = sum(
            cell["target"] != "hypoxemia_event" and cell["can_move"] and (
                assets.source_selector["cells"][cell["cell"]]["use_selector_interval"]
                or (
                    not assets.source_selector["cells"][cell["cell"]]["source_changed"]
                    and assets.intervals["cells"].get(cell["cell"], {}).get("runtime_eligible", False)
                )
            )
            for cell in result["cells"]
        )
        self.assertEqual(len(calibrated), expected)
        urine = next(cell for cell in result["cells"] if cell["cell"] == "urine_output@3h")
        self.assertEqual(urine["observed_value"], 26.0)
        self.assertTrue(urine["can_move"])
        if urine["interval_status"] == "externally_audited_mimic_split_conformal_90":
            self.assertLessEqual(urine["lower"], urine["forecast_point"])
            self.assertGreater(urine["upper"], urine["forecast_point"])
            self.assertGreaterEqual(urine["lower"], 0.0)
        else:
            self.assertIsNone(urine["lower"])
            self.assertIsNone(urine["upper"])
        assets = load_release(ROOT / "models")
        self.assertFalse(
            assets.intervals["cells"]["chloride@12h"]["runtime_eligible"]
        )

    def test_selected_source_and_interval_are_inseparable(self) -> None:
        assets = load_release(ROOT / "models")
        result = predict_patient((ROOT / "examples/patient_2.csv").read_text(), model_dir=ROOT / "models")
        for cell in result["cells"]:
            selected = assets.source_selector["cells"][cell["cell"]]
            self.assertEqual(cell["source_changed"], selected["source_changed"])
            if cell["can_move"] and cell["target"] != "hypoxemia_event":
                self.assertEqual(cell["source_model"], selected["source_model"])
                if selected["use_selector_interval"]:
                    if selected["interval_method"] == "normalized_current_predicted_delta":
                        from whole_body_jepa.runtime import _predict_interval_scale
                        features = np.asarray([[cell["observed_value"], abs(cell["forecast_point"] - cell["observed_value"])]])
                        expected = selected["normalized_radius"] * _predict_interval_scale(selected["scale_model"], features)
                    else:
                        expected = selected["radius"]
                    self.assertAlmostEqual(cell["upper"] - cell["forecast_point"], expected)
                    self.assertEqual(cell["interval_status"], "unified_source_disjoint_calibration_90")
                if selected["source_changed"]:
                    self.assertTrue(selected["validation"]["promotion_allowed"])
                    self.assertTrue(selected["use_selector_interval"])
        self.assertFalse(result["causal_claim_allowed"])

    def test_normalized_interval_scale_is_patient_conditional(self) -> None:
        from whole_body_jepa.runtime import _predict_interval_scale

        model = {
            "center": [0.0],
            "scale": [1.0],
            "beta": [0.0, 1.0],
            "lower": 0.1,
            "upper": 10.0,
        }
        low = _predict_interval_scale(model, np.asarray([[0.0]]))
        high = _predict_interval_scale(model, np.asarray([[1.0]]))
        self.assertGreater(high, low)

    def test_event_risk_abstains_without_o2sat_history(self) -> None:
        result = predict_patient(
            "hours_since_onset,heart_rate,map\n0,80,75\n3,84,72\n",
            model_dir=ROOT / "models",
        )
        events = [
            cell for cell in result["cells"] if cell["target"] == "hypoxemia_event"
        ]
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertTrue(event["abstained"])
            self.assertFalse(event["can_move"])
            self.assertIsNone(event["forecast_point"])
            self.assertIsNone(event["event_risk_probability"])
            self.assertEqual(event["interval_status"], "event_risk_abstained")

    def test_stale_continuous_target_abstains_instead_of_persisting(self) -> None:
        result = predict_patient(
            "hours_since_onset,map,glucose,heart_rate\n0,75,140,80\n12,,,82\n",
            model_dir=ROOT / "models",
        )
        glucose = next(cell for cell in result["cells"] if cell["cell"] == "glucose@1h")
        self.assertTrue(glucose["abstained"])
        self.assertFalse(glucose["can_move"])
        self.assertIsNone(glucose["forecast_point"])
        self.assertEqual(glucose["source_model"], "stale_target_abstained")
        self.assertEqual(glucose["fallback_reason"], "stale_target_anchor")

    def test_missing_target_uses_only_validated_cross_target_nowcast(self) -> None:
        result = predict_patient(
            "hours_since_onset,map,o2sat,respiratory_rate,temperature,glucose\n"
            "0,75,97,18,37.0,140\n3,72,96,20,37.3,132\n",
            model_dir=ROOT / "models",
        )
        bicarbonate = result["current_state"]["bicarbonate"]
        self.assertEqual(
            bicarbonate["source"], "validated_cross_target_nowcast"
        )
        self.assertIsNotNone(bicarbonate["value"])
        self.assertLessEqual(bicarbonate["lower"], bicarbonate["value"])
        self.assertGreaterEqual(bicarbonate["upper"], bicarbonate["value"])
        self.assertEqual(result["current_state"]["creatinine"]["source"], "missing")

    def test_observation_always_wins_over_nowcast(self) -> None:
        result = predict_patient(
            "hours_since_onset,bicarbonate,map,o2sat,respiratory_rate,temperature\n"
            "0,18,75,97,18,37.0\n3,20,72,96,20,37.3\n",
            model_dir=ROOT / "models",
        )
        bicarbonate = result["current_state"]["bicarbonate"]
        self.assertEqual(bicarbonate["source"], "observed")
        self.assertEqual(bicarbonate["value"], 20.0)

    def test_readme_covers_every_input_and_cell(self) -> None:
        readme = (ROOT / "README.md").read_text()
        assets = load_release(ROOT / "models")
        for variable in assets.checkpoint["variables"]:
            self.assertIn(f"| {variable} |", readme)
        registry = json.loads((ROOT / "models/runtime_registry.json").read_text())
        for cell in registry["cells"]:
            self.assertIn(f"| {cell['cell']} |", readme)


if __name__ == "__main__":
    unittest.main()
