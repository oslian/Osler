import unittest

from osler_jepa.observation_layer import (
    DENIED_AUTHORITIES,
    DERIVED_COMPLETION_RULES,
    SAME_GROUP_CALIBRATED_COMPLETION_GROUPS,
    observation_readiness,
    irregular_time_prediction_grid,
    resolve_irregular_time_forecast_cell,
    rollout_readiness,
    nowcast_state_grid,
    trajectory_prediction_grid,
    uncertainty_calibration_policy,
    whole_body_state_forecast_template,
    whole_body_observation_capabilities,
)


class ObservationLayerTests(unittest.TestCase):
    def test_validated_multihop_and_candidates_are_explicit(self):
        capabilities = {
            capability.name: capability
            for capability in whole_body_observation_capabilities()
        }

        self.assertTrue(capabilities["sepsis_map6_renal24_multihop"].is_validated)
        self.assertEqual(
            capabilities["sepsis_map6_renal24_multihop"].targets,
            ("creatinine", "bun"),
        )
        self.assertEqual(
            capabilities["sepsis_map6_renal24_multihop"].horizon_hours,
            (24,),
        )

        self.assertFalse(
            capabilities["respiratory_acid_base_observed_coupling"].is_validated
        )
        self.assertFalse(capabilities["sepsis_map6_renal48_multihop"].is_validated)

        nowcast = capabilities["whole_body_same_time_nowcasting"]
        self.assertTrue(nowcast.is_validated)
        self.assertEqual(nowcast.horizon_hours, (0,))
        self.assertIn("same_time_state_completion", nowcast.allowed_uses)
        self.assertIn("bilirubin_direct", nowcast.targets)

        mimiciv = capabilities["mimiciv_cross_database_observation_validation"]
        self.assertTrue(mimiciv.is_validated)
        self.assertEqual(mimiciv.scope, "external_database_observation_validation")
        self.assertEqual(mimiciv.horizon_hours, (0, 6))
        self.assertIn("14_target_calibrated_interval_cells", mimiciv.targets)

        full_mimic = capabilities["mimiciv_full_target_observation_coverage_6h"]
        self.assertTrue(full_mimic.is_validated)
        self.assertEqual(full_mimic.scope, "full_mimic_target_observation_coverage")
        self.assertEqual(full_mimic.horizon_hours, (0, 6))
        self.assertIn("19_validated_nowcast_targets", full_mimic.targets)
        self.assertIn("18_validated_6h_forecast_targets", full_mimic.targets)

        treatment_context = capabilities["mimiciv_unit_safe_treatment_precision"]
        self.assertTrue(treatment_context.is_validated)
        self.assertEqual(treatment_context.scope, "factual_observed_treatment_context")
        self.assertEqual(treatment_context.horizon_hours, (6,))
        self.assertIn("glucose", treatment_context.targets)
        self.assertIn("bicarbonate", treatment_context.targets)
        self.assertNotIn("potassium", treatment_context.targets)
        self.assertIn("not a treatment-effect claim", treatment_context.evidence)

        waveform = capabilities["mimiciv_waveform_precision_candidate"]
        self.assertFalse(waveform.is_validated)
        self.assertEqual(waveform.scope, "high_frequency_waveform_precision")
        self.assertIn("arterial_pressure_beat_to_beat", waveform.targets)
        self.assertIn("heart_rate", waveform.targets)
        self.assertIn("waveform_manifest_inspection", waveform.allowed_uses)
        self.assertNotIn("factual_prediction", waveform.allowed_uses)
        self.assertIn("57 targets", waveform.evidence)

        waveform_nowcast = capabilities["mimiciv_waveform_same_anchor_nowcast_validation"]
        self.assertTrue(waveform_nowcast.is_validated)
        self.assertEqual(waveform_nowcast.scope, "high_frequency_same_anchor_nowcast")
        self.assertEqual(waveform_nowcast.horizon_hours, (0,))
        self.assertIn("heart_rate", waveform_nowcast.targets)
        self.assertIn("waveform_same_anchor_nowcast", waveform_nowcast.allowed_uses)
        self.assertNotIn("future_forecast", waveform_nowcast.allowed_uses)

        waveform_sweep = capabilities["mimiciv_waveform_all_target_precision_sweep"]
        self.assertFalse(waveform_sweep.is_validated)
        self.assertIn("57_evaluated_targets", waveform_sweep.targets)
        self.assertIn("0_validated_waveform_increments", waveform_sweep.targets)
        self.assertNotIn("factual_prediction", waveform_sweep.allowed_uses)

        ed = capabilities["mimiciv_ed_scene_observation_validation"]
        self.assertTrue(ed.is_validated)
        self.assertEqual(ed.scope, "ed_scene_observation_validation")
        self.assertEqual(ed.horizon_hours, (0, 1, 3, 6))
        self.assertIn("6h_6_target_forecast_interval_cells", ed.targets)

        ed_to_icu = capabilities["mimiciv_ed_to_icu_baseline_candidate"]
        self.assertFalse(ed_to_icu.is_validated)
        self.assertEqual(ed_to_icu.scope, "ed_to_early_icu_baseline")
        self.assertEqual(ed_to_icu.horizon_hours, (1, 3, 6))
        self.assertIn("heart_rate_map_3h_6h_near_miss", ed_to_icu.targets)
        self.assertNotIn("factual_prediction", ed_to_icu.allowed_uses)

        nhanes = capabilities["nhanes_healthy_population_nowcast_validation"]
        self.assertTrue(nhanes.is_validated)
        self.assertEqual(nhanes.scope, "external_population_nowcast_validation")
        self.assertEqual(nhanes.horizon_hours, (0,))
        self.assertIn("29_cross_sectional_nowcast_targets", nhanes.targets)
        self.assertIn("hs_crp_fallback", nhanes.targets)
        self.assertIn("same_time_state_completion", nhanes.allowed_uses)
        self.assertNotIn("factual_prediction", nhanes.allowed_uses)
        self.assertIn("no forecast", nhanes.evidence)

        radiology_notes = capabilities["mimiciv_radiology_note_structured_observation_candidate"]
        self.assertFalse(radiology_notes.is_validated)
        self.assertEqual(radiology_notes.scope, "note_backed_measurement_depth")
        self.assertIn("rad_pleural_effusion", radiology_notes.targets)
        self.assertNotIn("factual_prediction", radiology_notes.allowed_uses)
        self.assertIn("structured_note_observation", radiology_notes.allowed_uses)

        neuro_notes = capabilities["eicu_neuro_note_structured_observation_candidate"]
        self.assertFalse(neuro_notes.is_validated)
        self.assertIn("neuro_gcs", neuro_notes.targets)
        self.assertIn("neuro_delirium_present", neuro_notes.targets)
        self.assertNotIn("same_time_state_completion", neuro_notes.allowed_uses)
        self.assertIn("EICU_NEURO_NOTE_ALL_ICU_FINDINGS.md", neuro_notes.evidence)
        self.assertIn("100,862 subjects", neuro_notes.evidence)

        cardiovascular_belief = capabilities["cardiovascular_heart_rate_belief_state"]
        self.assertTrue(cardiovascular_belief.is_validated)
        self.assertEqual(cardiovascular_belief.scope, "predict_update_belief_state")
        self.assertEqual(cardiovascular_belief.horizon_hours, (6,))
        self.assertIn("heart_rate", cardiovascular_belief.targets)
        self.assertIn("7/7 patient splits", cardiovascular_belief.evidence)
        self.assertIn("factual_prediction", cardiovascular_belief.allowed_uses)

        cardiovascular_remaining = capabilities["cardiovascular_belief_remaining_targets_candidate"]
        self.assertFalse(cardiovascular_remaining.is_validated)
        self.assertEqual(cardiovascular_remaining.scope, "predict_update_belief_state")
        self.assertIn("map", cardiovascular_remaining.targets)
        self.assertNotIn("heart_rate", cardiovascular_remaining.targets)
        self.assertNotIn("factual_prediction", cardiovascular_remaining.allowed_uses)

        electrolyte_belief = capabilities["electrolyte_acid_base_belief_state"]
        self.assertTrue(electrolyte_belief.is_validated)
        self.assertEqual(electrolyte_belief.scope, "predict_update_belief_state")
        self.assertEqual(electrolyte_belief.horizon_hours, (6,))
        self.assertIn("potassium", electrolyte_belief.targets)
        self.assertIn("bicarbonate", electrolyte_belief.targets)
        self.assertIn("anion_gap", electrolyte_belief.targets)
        self.assertIn("7/7 patient", electrolyte_belief.evidence)
        self.assertIn("factual_prediction", electrolyte_belief.allowed_uses)

        electrolyte_remaining = capabilities["electrolyte_belief_remaining_targets_candidate"]
        self.assertFalse(electrolyte_remaining.is_validated)
        self.assertIn("sodium_near_miss", electrolyte_remaining.targets)
        self.assertIn("magnesium_hospital_fail", electrolyte_remaining.targets)
        self.assertNotIn("factual_prediction", electrolyte_remaining.allowed_uses)

        respiratory_belief = capabilities["respiratory_gas_exchange_belief_state"]
        self.assertTrue(respiratory_belief.is_validated)
        self.assertEqual(respiratory_belief.scope, "predict_update_belief_state")
        self.assertEqual(respiratory_belief.horizon_hours, (6,))
        self.assertIn("o2sat", respiratory_belief.targets)
        self.assertIn("respiratory_rate", respiratory_belief.targets)
        self.assertIn("bicarbonate", respiratory_belief.targets)
        self.assertIn("7/7 patient", respiratory_belief.evidence)
        self.assertIn("factual_prediction", respiratory_belief.allowed_uses)

        respiratory_remaining = capabilities["respiratory_belief_remaining_targets_candidate"]
        self.assertFalse(respiratory_remaining.is_validated)
        self.assertIn("map_near_miss", respiratory_remaining.targets)
        self.assertIn("paco2", respiratory_remaining.targets)
        self.assertIn("ph", respiratory_remaining.targets)
        self.assertNotIn("factual_prediction", respiratory_remaining.allowed_uses)

        endocrine_belief = capabilities["endocrine_glycemic_stress_belief_state"]
        self.assertTrue(endocrine_belief.is_validated)
        self.assertEqual(endocrine_belief.scope, "predict_update_belief_state")
        self.assertEqual(endocrine_belief.horizon_hours, (6,))
        self.assertIn("glucose", endocrine_belief.targets)
        self.assertIn("anion_gap", endocrine_belief.targets)
        self.assertIn("bicarbonate", endocrine_belief.targets)
        self.assertIn("7/7 patient", endocrine_belief.evidence)
        self.assertIn("factual_prediction", endocrine_belief.allowed_uses)

        endocrine_remaining = capabilities["endocrine_belief_remaining_targets_candidate"]
        self.assertFalse(endocrine_remaining.is_validated)
        self.assertIn("serum_ketones_sparse", endocrine_remaining.targets)
        self.assertIn("tsh_sparse", endocrine_remaining.targets)
        self.assertIn("cortisol_sparse", endocrine_remaining.targets)
        self.assertNotIn("factual_prediction", endocrine_remaining.allowed_uses)

        hormone_axes = capabilities["endocrine_hormone_axis_belief_candidate"]
        self.assertFalse(hormone_axes.is_validated)
        self.assertIn("water_balance_adh_like_drive", hormone_axes.targets)
        self.assertIn("insulin_response_state", hormone_axes.targets)
        self.assertNotIn("factual_prediction", hormone_axes.allowed_uses)

        nonlinear_rollout = capabilities["whole_body_nonlinear_latent_rollout_candidate"]
        self.assertFalse(nonlinear_rollout.is_validated)
        self.assertEqual(nonlinear_rollout.horizon_hours, (1, 3, 6, 12, 24, 48))
        self.assertIn("nonlinear_state_transition", nonlinear_rollout.targets)
        self.assertNotIn("factual_prediction", nonlinear_rollout.allowed_uses)

        hormone_labs = capabilities["mimiciv_hormone_lab_targets_candidate"]
        self.assertFalse(hormone_labs.is_validated)
        self.assertEqual(hormone_labs.scope, "sparse_hormone_observation_targets")
        self.assertIn("tsh", hormone_labs.targets)
        self.assertIn("free_t4", hormone_labs.targets)
        self.assertIn("cortisol", hormone_labs.targets)
        self.assertNotIn("factual_prediction", hormone_labs.allowed_uses)

        coupled_sparse = capabilities["mimiciv_sparse_lab_coupled_irregular_candidate"]
        self.assertFalse(coupled_sparse.is_validated)
        self.assertIn("troponin_t", coupled_sparse.targets)
        self.assertIn("no_promoted_coupled_sparse_lab_cell", coupled_sparse.targets)
        self.assertIn("0/21", coupled_sparse.evidence)
        self.assertNotIn("factual_prediction", coupled_sparse.allowed_uses)

    def test_readiness_is_observation_only_and_fail_closed(self):
        readiness = observation_readiness()

        self.assertGreaterEqual(readiness["validated_count"], 8)
        self.assertGreaterEqual(readiness["candidate_count"], 1)

        boundary = readiness["safety_boundary"]
        self.assertTrue(boundary["observation_and_factual_prediction_allowed"])
        self.assertFalse(boundary["causal_claim_allowed"])
        self.assertFalse(boundary["counterfactual_claim_allowed"])
        self.assertFalse(boundary["clinical_claim_allowed"])
        self.assertFalse(boundary["runtime_decision_authority"])
        self.assertFalse(boundary["checkpoint_promotion_allowed"])
        self.assertFalse(boundary["active_rule_promotion_allowed"])

        for capability in readiness["validated_capabilities"]:
            self.assertTrue(
                "factual_prediction" in capability["allowed_uses"]
                or "same_time_state_completion" in capability["allowed_uses"]
            )
            for authority in DENIED_AUTHORITIES:
                self.assertIn(authority, capability["denied_authorities"])

    def test_trajectory_grid_moves_only_validated_target_horizons(self):
        cells = {
            (cell.target, cell.horizon_hours): cell
            for cell in trajectory_prediction_grid()
        }

        self.assertEqual(cells[("glucose", 6)].status, "validated")
        self.assertTrue(cells[("glucose", 6)].can_move)
        self.assertEqual(cells[("glucose", 1)].source, "persistence")
        self.assertFalse(cells[("glucose", 1)].can_move)

        self.assertEqual(cells[("creatinine", 24)].status, "validated")
        self.assertEqual(cells[("creatinine", 48)].status, "validated")
        self.assertFalse(cells[("creatinine", 6)].can_move)
        self.assertEqual(cells[("creatinine", 6)].status, "fallback_wrong_horizon")

        self.assertFalse(cells[("bilirubin", 24)].can_move)
        self.assertEqual(
            cells[("bilirubin", 24)].status,
            "fallback_observability_limited",
        )

        irregular = irregular_time_prediction_grid()
        self.assertIn(
            ("inr", (6.0, 24.0)),
            {(cell["target"], tuple(cell["time_window_hours"])) for cell in irregular},
        )
        self.assertTrue(all(cell["can_move"] for cell in irregular))

        inr_12h = resolve_irregular_time_forecast_cell("INR", 12)
        self.assertEqual(inr_12h["status"], "validated")
        self.assertEqual(inr_12h["time_window_hours"], (6.0, 24.0))
        self.assertEqual(inr_12h["interval_status"], "calibrated")
        self.assertEqual(inr_12h["interval_level"], 0.90)
        self.assertTrue(inr_12h["can_move"])
        self.assertIsNone(inr_12h["point_estimate"])

        inr_36h = resolve_irregular_time_forecast_cell("inr", 36)
        self.assertEqual(inr_36h["time_window_hours"], (24.0, 72.0))
        self.assertEqual(inr_36h["interval_status"], "calibrated")

        ptt_3h = resolve_irregular_time_forecast_cell("ptt", 3)
        self.assertEqual(ptt_3h["status"], "validated")
        self.assertEqual(ptt_3h["interval_status"], "needs_calibration_audit")
        self.assertIsNone(ptt_3h["interval_level"])
        self.assertTrue(ptt_3h["can_move"])

        ptt_6h = resolve_irregular_time_forecast_cell("ptt", 6)
        self.assertEqual(ptt_6h["time_window_hours"], (0.0, 6.0))
        self.assertEqual(ptt_6h["query_boundary_policy"], "right_closed_irregular_bins")

        unsupported = resolve_irregular_time_forecast_cell("cortisol", 12)
        self.assertEqual(unsupported["status"], "fallback")
        self.assertFalse(unsupported["can_move"])
        self.assertEqual(unsupported["source"], "persistence_or_missing")
        self.assertEqual(unsupported["interval_status"], "not_available")

    def test_whole_body_state_forecast_template_unifies_three_objects(self):
        nowcasts = nowcast_state_grid()
        self.assertEqual(len(nowcasts), 76)
        self.assertTrue(all(cell.time_axis == "current" for cell in nowcasts))
        self.assertTrue(all(cell.can_estimate for cell in nowcasts))
        self.assertTrue(all(not cell.can_move for cell in nowcasts))
        self.assertIn(
            ("hepatic_failure", "bilirubin_direct"),
            {(cell.module, cell.target) for cell in nowcasts},
        )
        self.assertIn(
            ("mimiciv_full_target_coverage", "serum_osmolality"),
            {(cell.module, cell.target) for cell in nowcasts},
        )

        template = whole_body_state_forecast_template()
        self.assertEqual(template["object_name"], "whole_body_state_forecast")
        self.assertFalse(template["runtime_values_included"])
        self.assertEqual(
            template["current_state"]["completion_source_hierarchy"],
            [
                "observed",
                "derived_formula_completion",
                "same_group_calibrated_completion",
                "same_time_nowcast_ridge",
                "missing",
            ],
        )
        self.assertIn("map", DERIVED_COMPLETION_RULES)
        self.assertIn("red_cell_indices", SAME_GROUP_CALIBRATED_COMPLETION_GROUPS)
        self.assertTrue(
            template["current_state"]["policy"]["same_group_calibration_not_counted_as_cross_system_nowcast"]
        )
        self.assertEqual(
            template["current_state"]["validated_nowcast_module_target_cells"],
            76,
        )
        self.assertGreaterEqual(
            template["future_trajectory"]["validated_cell_count"],
            154,
        )
        future_cells = template["future_trajectory"]["cells"]
        self.assertIn(
            ("respiratory", "paco2", 3),
            {
                (cell["module"], cell["target"], cell["horizon_hours"])
                for cell in future_cells
                if cell["status"] == "validated"
            },
        )
        self.assertIn(
            ("sepsis", "platelets", 6),
            {
                (cell["module"], cell["target"], cell["horizon_hours"])
                for cell in future_cells
                if cell["status"] == "validated"
            },
        )
        self.assertIn(
            ("mimiciv_full_target_coverage", "pao2", 6),
            {
                (cell["module"], cell["target"], cell["horizon_hours"])
                for cell in future_cells
                if cell["status"] == "validated"
            },
        )
        self.assertIn(
            ("sepsis", "platelets", 6),
            {
                (cell["module"], cell["target"], cell["horizon_hours"])
                for cell in future_cells
                if cell["interval_status"] == "calibrated"
            },
        )
        self.assertFalse(template["safety_boundary"]["causal_claim_allowed"])
        self.assertFalse(template["safety_boundary"]["clinical_claim_allowed"])
        self.assertFalse(template["safety_boundary"]["runtime_decision_authority"])
        self.assertIn(
            "mimiciv_cross_database_coverage_audit.json",
            template["source_artifacts"],
        )
        self.assertEqual(template["future_trajectory"]["irregular_time_validated_cell_count"], 5)
        self.assertIn(
            ("inr", (6.0, 24.0)),
            {
                (cell["target"], tuple(cell["time_window_hours"]))
                for cell in template["future_trajectory"]["irregular_time_cells"]
            },
        )
        self.assertIn(
            "mimiciv_ed_observation_coverage_audit_6h.json",
            template["source_artifacts"],
        )
        self.assertIn(
            "mimiciv_ed_to_icu_baseline_audit_6h.json",
            template["source_artifacts"],
        )
        self.assertIn(
            "nhanes_2017_2018_nowcast_audit.json",
            template["source_artifacts"],
        )
        self.assertIn(
            "MIMICIV_TREATMENT_CONTEXT_ACCURACY_FINDINGS.md",
            template["source_artifacts"],
        )
        self.assertIn(
            "MIMICIV_HORMONE_DATA_FINDINGS.md",
            template["source_artifacts"],
        )
        self.assertIn(
            "MIMICIV_SPARSE_LAB_COUPLED_IRREGULAR_FINDINGS.md",
            template["source_artifacts"],
        )
        self.assertIn(
            "NONLINEAR_LATENT_ROLLOUT_COMPREHENSIVE_AUDIT.md",
            template["source_artifacts"],
        )
        self.assertIn(
            "NONLINEAR_LATENT_ROLLOUT_BODY_SYSTEM_SCREEN.md",
            template["source_artifacts"],
        )
        self.assertIn(
            "NONLINEAR_LATENT_ROLLOUT_REGIME_UNCERTAINTY_FINDINGS.md",
            template["source_artifacts"],
        )
        self.assertIn(
            "NONLINEAR_LATENT_ROLLOUT_REGIME_BODY_SCREEN.md",
            template["source_artifacts"],
        )
        self.assertIn(
            "MIMICIV_WAVEFORM_PRECISION_FINDINGS.md",
            template["source_artifacts"],
        )
        self.assertIn(
            "eicu_electrolyte_belief_full_audit.json",
            template["source_artifacts"],
        )
        self.assertIn(
            "eicu_respiratory_belief_full_audit.json",
            template["source_artifacts"],
        )
        self.assertIn(
            "eicu_endocrine_belief_full_audit.json",
            template["source_artifacts"],
        )

    def test_uncertainty_policy_requires_calibration_before_intervals(self):
        policy = uncertainty_calibration_policy()

        self.assertEqual(policy["primary_coverage_target"], 0.9)
        self.assertTrue(policy["coverage_gate"]["required"])
        self.assertTrue(policy["coverage_gate"]["hospital_heldout_required"])
        self.assertIn("needs_calibration_audit", policy["missing_calibration_policy"])

        readiness = rollout_readiness()
        self.assertIn(1, readiness["horizons_hours"])
        self.assertIn(48, readiness["horizons_hours"])
        self.assertGreater(readiness["validated_cell_count"], 0)
        self.assertGreater(readiness["fallback_or_closed_cell_count"], 0)
        self.assertTrue(readiness["trajectory_policy"]["no_recursive_unvalidated_rollout"])


if __name__ == "__main__":
    unittest.main()
