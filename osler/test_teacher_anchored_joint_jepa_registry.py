import unittest
from pathlib import Path

from teacher_anchored_joint_jepa_registry import (
    build_registry,
    select_validated_coupling_path,
)
from teacher_anchored_joint_jepa_external_gate import (
    _external_pass,
    _patient_candidate_paths,
    _patient_candidates,
)
from whole_body_joint_runtime import forecast_joint_values


class TeacherAnchoredJointJEPARegistryTests(unittest.TestCase):
    def test_external_candidates_require_patient_conformal(self):
        bounded = {
            "aggregate": {
                "adjacent_three_level": {
                    "passes": {
                        "evaluated_splits": 7,
                        "beats_teacher_splits": 7,
                        "beats_persistence_splits": 7,
                        "conformal_pass_splits": 7,
                        "beats_no_route_splits": 7,
                        "direct_organ_beats_system_only_splits": 7,
                        "median_direct_organ_delta_vs_system_only": -0.02,
                    },
                    "point_only": {
                        "evaluated_splits": 7,
                        "beats_teacher_splits": 7,
                        "beats_persistence_splits": 7,
                        "conformal_pass_splits": 6,
                        "beats_no_route_splits": 7,
                        "direct_organ_beats_system_only_splits": 7,
                        "median_direct_organ_delta_vs_system_only": -0.02,
                    },
                }
            }
        }
        self.assertEqual(_patient_candidates(bounded), {"passes"})
        self.assertEqual(
            _patient_candidate_paths(bounded)["passes"]["selected_path"],
            "direct_organ",
        )

    def test_external_pass_uses_selected_path_against_system_only(self):
        report = {
            "values": {
                "creatinine@12h": {
                    "teacher_bootstrap": {"pass": True},
                    "persistence_bootstrap": {"pass": True},
                }
            },
            "conformal": {"creatinine@12h": {"pass": True}},
            "module_balanced": {"creatinine@12h": {"pass": True}},
            "no_route_ablation": {
                "creatinine@12h": {
                    "no_route_bootstrap": {"pass": True}
                }
            },
            "selected_coupling_ablation": {
                "creatinine@12h": {
                    "system_only_bootstrap": {"pass": True}
                }
            },
        }
        result = _external_pass(report, "creatinine@12h")
        self.assertTrue(result["selected_coupling_bootstrap"])
        self.assertNotIn("no_organ_coupling_bootstrap", result)

    def _payloads(self):
        key = "map@3h"
        bounded = {
            "architecture": {
                "domain_invariance_weight": 0.10,
                "domain_invariance_columns": ["hospitalid", "careunit"],
            },
            "aggregate": {
                "adjacent_three_level": {
                    key: {
                        "evaluated_splits": 7,
                        "beats_teacher_splits": 7,
                        "beats_persistence_splits": 7,
                        "beats_no_route_splits": 7,
                        "conformal_pass_splits": 7,
                        "direct_organ_beats_system_only_splits": 7,
                        "median_direct_organ_delta_vs_system_only": -0.01,
                        "body_mediated_beats_system_only_splits": 7,
                        "median_body_mediated_delta_vs_system_only": -0.03,
                    }
                }
            }
        }
        external = {
            "registry": {
                key: {
                    "requirements": {
                        "hospital_conformal": True,
                        "care_unit_conformal": True,
                        "time_conformal": True,
                        "hospital_bootstrap": True,
                    }
                }
            }
        }
        scoring = {
            "reports": [
                {
                    "seed": seed,
                    "latent_svd": {"pass": True},
                    "variants": {
                        "adjacent_three_level": {
                            "cells": {
                                key: {
                                    "direction_accuracy": 0.60,
                                    "delta_correlation": 0.20,
                                    "teacher_bootstrap": {"pass": True},
                                    "persistence_bootstrap": {"pass": True},
                                }
                            },
                            "conformal": {key: {"pass": True}},
                        }
                    },
                }
                for seed in range(7)
            ]
        }
        return bounded, external, scoring

    def test_complete_evidence_validates_only_the_cell(self):
        output = build_registry(*self._payloads())
        self.assertEqual(
            output["validated_target_horizon_cells"], ["map@3h"]
        )
        self.assertTrue(
            output["target_horizon_registry"]["map@3h"][
                "cross_organ_claim_allowed"
            ]
        )
        self.assertEqual(
            output["target_horizon_registry"]["map@3h"][
                "coupling_path"
            ]["selected_path"],
            "body_mediated",
        )
        self.assertTrue(
            output["target_horizon_registry"]["map@3h"][
                "coupling_path"
            ]["disable_organ_coupling"]
        )
        self.assertEqual(
            output["candidate_configuration"]["domain_invariance_weight"],
            0.10,
        )
        self.assertEqual(
            output["candidate_configuration"]["domain_invariance_columns"],
            ["hospitalid", "careunit"],
        )

    def test_coupling_path_requires_independent_system_only_gain(self):
        self.assertIsNone(
            select_validated_coupling_path(
                {
                    "beats_no_organ_coupling_splits": 7,
                    "median_delta_vs_no_organ_coupling": -0.1,
                }
            )
        )

    def test_coupling_path_chooses_stronger_validated_path(self):
        selected = select_validated_coupling_path(
            {
                "direct_organ_beats_system_only_splits": 7,
                "median_direct_organ_delta_vs_system_only": -0.01,
                "body_mediated_beats_system_only_splits": 7,
                "median_body_mediated_delta_vs_system_only": -0.03,
            }
        )
        self.assertEqual(selected["selected_path"], "body_mediated")
        self.assertTrue(selected["disable_organ_coupling"])
        self.assertFalse(selected["disable_body_feedback"])

    def test_requested_path_must_pass_independently(self):
        cell = {
            "direct_organ_beats_system_only_splits": 7,
            "median_direct_organ_delta_vs_system_only": -0.01,
            "body_mediated_beats_system_only_splits": 6,
            "median_body_mediated_delta_vs_system_only": -0.03,
        }
        selected = select_validated_coupling_path(
            cell, requested_path="direct_organ"
        )
        self.assertEqual(selected["selected_path"], "direct_organ")
        self.assertIsNone(
            select_validated_coupling_path(
                cell, requested_path="body_mediated"
            )
        )

    def test_registry_records_input_contract_and_units(self):
        output = build_registry(
            *self._payloads(),
            input_contract_id="joint_interval_urine.v1",
            variable_units={"urine_output": "mL/hour"},
            candidate_id="aligned_treatment_interval_urine_jepa",
        )
        self.assertEqual(
            output["candidate_id"],
            "aligned_treatment_interval_urine_jepa",
        )
        self.assertEqual(output["input_contract_id"], "joint_interval_urine.v1")
        self.assertEqual(output["variable_units"]["urine_output"], "mL/hour")

    def test_missing_split_fails_closed(self):
        bounded, external, scoring = self._payloads()
        scoring["reports"].pop()
        output = build_registry(bounded, external, scoring)
        self.assertEqual(output["validated_target_horizon_cells"], [])

    def test_legacy_variant_cannot_write_promotion_registry(self):
        bounded, external, scoring = self._payloads()
        with self.assertRaises(ValueError):
            build_registry(
                bounded,
                external,
                scoring,
                variant="sparse_distributional",
            )

    def test_materialized_registry_moves_only_full_cohort_cells(self):
        registry = Path(
            "teacher_anchored_joint_jepa_validated_registry_20260731.json"
        )
        self.assertTrue(registry.exists())
        result_6h = forecast_joint_values(
            prediction=[[1.7, 90.0, 80.0]],
            current=[[1.6, 75.0, 70.0]],
            variables=("creatinine", "heart_rate", "map"),
            horizons=6.0,
            registry=registry,
            checkpoint_status=(
                "validated_target_gated_joint_research_only"
            ),
            input_contract_id="joint_asof_whole_body_state.v3",
            variable_units={
                "creatinine": "mg/dL",
                "heart_rate": "beats/min",
                "map": "mmHg",
            },
        )
        self.assertTrue(
            result_6h["forecasts"][0]["creatinine"]["can_move"]
        )
        self.assertFalse(
            result_6h["forecasts"][0]["heart_rate"]["can_move"]
        )
        self.assertFalse(result_6h["forecasts"][0]["map"]["can_move"])
        result_12h = forecast_joint_values(
            prediction=[[1.8, 90.0]],
            current=[[1.6, 75.0]],
            variables=("creatinine", "heart_rate"),
            horizons=12.0,
            registry=registry,
            checkpoint_status=(
                "validated_target_gated_joint_research_only"
            ),
            input_contract_id="joint_asof_whole_body_state.v3",
            variable_units={
                "creatinine": "mg/dL",
                "heart_rate": "beats/min",
            },
        )
        self.assertTrue(
            result_12h["forecasts"][0]["creatinine"]["can_move"]
        )
        self.assertFalse(
            result_12h["forecasts"][0]["heart_rate"]["can_move"]
        )
        result_3h = forecast_joint_values(
            prediction=[[1.7, 85.0]],
            current=[[1.6, 75.0]],
            variables=("creatinine", "heart_rate"),
            horizons=3.0,
            registry=registry,
            checkpoint_status=(
                "validated_target_gated_joint_research_only"
            ),
            input_contract_id="joint_asof_whole_body_state.v3",
            variable_units={
                "creatinine": "mg/dL",
                "heart_rate": "beats/min",
            },
        )
        self.assertFalse(
            result_3h["forecasts"][0]["creatinine"]["can_move"]
        )
        self.assertTrue(
            result_3h["forecasts"][0]["heart_rate"]["can_move"]
        )


if __name__ == "__main__":
    unittest.main()
