import unittest

import numpy as np
import pandas as pd
import torch

from nonlinear_latent_rollout import StateScaler
from pure_jepa_runtime import forecast_history
from pure_jepa_body_system import PureJEPABodyModel
from sparse_aware_full_gate import build_registry
from sparse_aware_pure_jepa import SparseAwarePureJEPABodyModel
from train_pure_jepa_dka_heads import PureJEPADKATargetHeads
from whole_body_forecast_contract import forecast_whole_body
from next_measurement_time import next_measurement_labels, unavailable_next_measurement_labels
from whole_body_shared_jepa import SharedWholeBodySparseJEPA


class PureJEPAModelTests(unittest.TestCase):
    def test_body_model_has_target_specific_heads_and_finite_output(self):
        model = PureJEPABodyModel(n_state=4, input_dim=13, hidden=16, latent=8)
        observations = torch.randn(3, 5, 13)
        current = torch.randn(3, 4)
        horizon = torch.tensor([1.0, 6.0, 24.0])
        output = model(observations, current, horizon)
        self.assertEqual(len(model.target_heads), 4)
        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_dka_head_model_has_independent_heads(self):
        model = PureJEPADKATargetHeads(feature_dim=12, target_count=3, hidden=16)
        features = torch.randn(5, 12)
        current = torch.randn(5, 3)
        output = model(features, current)
        self.assertEqual(len(model.target_heads), 3)
        self.assertEqual(tuple(output.shape), (5, 3))
        self.assertTrue(torch.isfinite(output).all())

    def test_runtime_falls_back_for_unvalidated_target(self):
        variables = ("map", "creatinine")
        scaler = StateScaler(
            variables,
            np.asarray([70.0, 1.0], dtype=np.float32),
            np.asarray([10.0, 0.5], dtype=np.float32),
        )
        model = PureJEPABodyModel(n_state=2, input_dim=7, hidden=8, latent=6)
        history = pd.DataFrame(
            [
                {"map_t": 65.0, "creatinine_t": 1.1, "map_age_hr": 0.0, "creatinine_age_hr": 2.0, "hours_since_onset": 0.0},
                {"map_t": 70.0, "creatinine_t": 1.2, "map_age_hr": 0.0, "creatinine_age_hr": 1.0, "hours_since_onset": 3.0},
            ]
        )
        registry = {
            "modules": {
                "aki": {
                    "map": {"can_move": True, "status": "validated_move", "rejection_reasons": []},
                    "creatinine": {"can_move": False, "status": "fallback_persistence", "rejection_reasons": ["conformal_gate"]},
                }
            }
        }
        result = forecast_history(
            history,
            (
                model,
                scaler,
                {
                    "promotion_status": "validated",
                    "architecture": {"measurement_pure": True},
                },
            ),
            module="aki",
            horizon_hours=6,
            registry=registry,
        )
        self.assertEqual(result["variables"]["map"]["source"], "pure_jepa_validated")
        self.assertEqual(result["variables"]["creatinine"]["source"], "persistence_fallback")
        self.assertAlmostEqual(result["variables"]["creatinine"]["point"], 1.2, places=5)
        self.assertFalse(result["causal_claim_allowed"])
        candidate_result = forecast_history(
            history,
            (model, scaler, {"promotion_status": "candidate_only", "architecture": {}}),
            module="aki",
            horizon_hours=6,
            registry=registry,
        )
        self.assertEqual(candidate_result["variables"]["map"]["source"], "persistence_fallback")
        self.assertIn("checkpoint_candidate_only", candidate_result["variables"]["map"]["rejection_reasons"])

    def test_sparse_model_exposes_observation_uncertainty_and_stability_heads(self):
        model = SparseAwarePureJEPABodyModel(n_state=3, input_dim=10, hidden=16, latent=8)
        result = model(
            torch.randn(4, 5, 10),
            torch.randn(4, 3),
            torch.tensor([1.0, 3.0, 12.0, 48.0]),
        )
        self.assertEqual(set(result), {"value", "observation_logit", "log_scale", "stable_logit"})
        for value in result.values():
            self.assertEqual(tuple(value.shape), (4, 3))
        self.assertTrue(torch.isfinite(value).all())

    def test_sparse_model_can_expose_exact_measurement_time_head(self):
        model = SparseAwarePureJEPABodyModel(
            n_state=2, input_dim=8, hidden=16, latent=8, measurement_time_head=True
        )
        result = model(torch.randn(2, 5, 8), torch.randn(2, 2), torch.tensor([1.0, 6.0]))
        self.assertIn("next_measurement_time_log", result)
        self.assertTrue(torch.isfinite(result["next_measurement_time_log"]).all())

    def test_runtime_accepts_sparse_checkpoint_output(self):
        variables = ("troponin", "inr")
        scaler = StateScaler(
            variables,
            np.asarray([0.1, 1.0], dtype=np.float32),
            np.asarray([0.1, 0.5], dtype=np.float32),
        )
        model = SparseAwarePureJEPABodyModel(n_state=2, input_dim=7, hidden=8, latent=6)
        history = pd.DataFrame(
            [{"troponin_t": 0.2, "inr_t": 1.2, "troponin_age_hr": 1.0, "inr_age_hr": 2.0, "hours_since_onset": 0.0}]
        )
        registry = {"modules": {"cardiac": {"troponin": {"can_move": False, "status": "fallback_persistence", "rejection_reasons": ["patient_gate"]}, "inr": {"can_move": False, "status": "fallback_persistence", "rejection_reasons": ["conformal_gate"]}}}}
        result = forecast_history(
            history,
            (model, scaler, {"model_type": "SparseAwarePureJEPABodyModel", "architecture": {}}),
            module="cardiac",
            horizon_hours=6,
            registry=registry,
        )
        self.assertIn("observation_probability", result["variables"]["inr"])
        self.assertIn("stable_probability", result["variables"]["inr"])
        self.assertEqual(result["variables"]["troponin"]["source"], "persistence_fallback")

    def test_sparse_gate_registry_never_moves_without_retrained_checkpoint(self):
        registry = build_registry(
            {
                "aki": {
                    "aggregate": {
                        "creatinine": {
                            "patient_gate": True,
                            "hospital_gate": True,
                            "conformal_gate": True,
                            "validated_full_gate": True,
                        }
                    }
                }
            }
        )
        entry = registry["modules"]["aki"]["creatinine"]
        self.assertTrue(entry["promotion_ready"])
        self.assertFalse(entry["can_move"])
        self.assertEqual(entry["status"], "promotion_ready_checkpoint_retrain_required")

    def test_runtime_requires_requested_horizon_interval_artifact(self):
        variables = ("map",)
        scaler = StateScaler(
            variables, np.asarray([70.0], dtype=np.float32), np.asarray([10.0], dtype=np.float32)
        )
        model = SparseAwarePureJEPABodyModel(n_state=1, input_dim=4, hidden=8, latent=6)
        history = pd.DataFrame([
            {"map_t": 70.0, "map_age_hr": 0.0, "hours_since_onset": 0.0}
        ])
        metadata = {
            "model_type": "SparseAwarePureJEPABodyModel",
            "promotion_status": "validated_research_only",
            "validated_targets": ["map"],
            "conformal": {
                "map": {
                    "status": "PASS", "scope": "target_by_horizon",
                    "by_horizon": {"6": {"status": "PASS", "q90": 2.0}},
                }
            },
            "architecture": {},
        }
        registry = {"modules": {"aki": {"map": {"can_move": True, "status": "validated_move", "rejection_reasons": []}}}}
        result_6 = forecast_history(history, (model, scaler, metadata), module="aki", horizon_hours=6, registry=registry)
        result_3 = forecast_history(history, (model, scaler, metadata), module="aki", horizon_hours=3, registry=registry)
        self.assertEqual(result_6["variables"]["map"]["interval_status"], "calibrated")
        self.assertEqual(result_3["variables"]["map"]["interval_status"], "needs_horizon_calibration_artifact")

    def test_whole_body_contract_fails_closed_without_history(self):
        output = forecast_whole_body(
            histories={},
            checkpoints={"aki": "/missing/aki.pt"},
            registry={"modules": {}},
            horizons=(6,),
        )
        self.assertEqual(output["schema"], "whole_body_state_forecast.v2")
        self.assertEqual(output["modules"]["aki"]["status"], "missing_history")
        self.assertFalse(output["semantics"]["shared_latent_validated"])

    def test_next_measurement_labels_use_strict_future_event(self):
        anchors = pd.DataFrame({"subject_id": [1, 1], "t": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 02:00"])})
        events = pd.DataFrame({"subject_id": [1, 1], "charttime": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 03:00"]), "value": [1.0, 2.0]})
        result = next_measurement_labels(anchors, events, value_col="value")
        self.assertAlmostEqual(result.iloc[0]["next_measurement_time_hr"], 3.0, places=5)
        self.assertAlmostEqual(result.iloc[1]["next_measurement_time_hr"], 1.0, places=5)
        unavailable = unavailable_next_measurement_labels(2, "raw_events_not_present")
        self.assertTrue(unavailable["next_measurement_time_hr"].isna().all())

    def test_shared_whole_body_model_has_one_encoder_and_module_conditioning(self):
        model = SharedWholeBodySparseJEPA(n_state=4, input_dim=13, module_count=3, hidden=8, latent=6)
        output = model(torch.randn(2, 5, 13), torch.randn(2, 4), torch.tensor([1.0, 6.0]), torch.tensor([0, 2]))
        self.assertEqual(tuple(output["value"].shape), (2, 4))
        self.assertEqual(tuple(output["observation_logit"].shape), (2, 4))
        self.assertTrue(torch.isfinite(output["value"]).all())


if __name__ == "__main__":
    unittest.main()
