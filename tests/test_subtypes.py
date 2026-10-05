from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from scripts.validate_subtype_external import eicu_path
from scripts.validate_trajectory_subtype_eicu import (
    FUTURE_EVENT_POLICY,
    _apply_selective_gate,
    _causal_patient_residual_update,
    _clinical_event_matrix,
    _choose_gate_threshold,
    _consensus_correction,
    _select_group_robust_stack,
    _selective_gate_features,
    _treatment_covariates,
)

from whole_body_jepa.subtypes import (
    SoftPhysiologySubtypeModel,
    _patient_equal_weights,
    causal_history_covariates,
    causal_history_covariates_at_anchors,
    causal_trajectory_features,
    subtype_features,
)


def _features(group: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(0.0, 0.15, size=(len(group), 311))
    matrix[:, 0] += np.where(group == 0, -3.0, 3.0)
    matrix[:, 64] += np.where(group == 0, -2.0, 2.0)
    matrix[:, 257] += np.where(group == 0, -1.0, 1.0)
    return matrix


class SoftPhysiologySubtypeTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(9)
        self.group = np.repeat([0, 1], 60)
        self.features = _features(self.group, rng)
        self.subjects = np.asarray([f"patient-{index}" for index in range(120)])
        self.base = np.zeros((120, 1), dtype=np.float64)
        self.targets = np.where(self.group[:, None] == 0, -2.0, 2.0)
        self.model = SoftPhysiologySubtypeModel(
            subtype_count=2,
            min_effective_subjects=10,
            subtype_prior_strength=1.0,
            patient_prior_strength=4.0,
            random_state=4,
        ).fit(
            self.features,
            base_points=self.base,
            observed_targets=self.targets,
            observed_mask=np.ones_like(self.targets, dtype=bool),
            subject_ids=self.subjects,
            cell_keys=("glucose@6h",),
        )

    def test_router_uses_only_target_independent_pre_anchor_blocks(self) -> None:
        selected = subtype_features(self.features)
        self.assertEqual(selected.shape, (120, 213))
        altered = self.features.copy()
        altered[:, 160:257] += 1000.0
        altered[:, 310] += 1000.0
        np.testing.assert_allclose(subtype_features(altered), selected)

    def test_router_appends_measurement_pure_phenotype_covariates(self) -> None:
        covariates = np.arange(360, dtype=np.float64).reshape(120, 3)
        selected = subtype_features(self.features, covariates)
        self.assertEqual(selected.shape, (120, 216))
        np.testing.assert_allclose(selected[:, -3:], covariates)

    def test_soft_subtypes_recover_distinct_physiology(self) -> None:
        probability = self.model.responsibilities(self.features)
        self.assertEqual(probability.shape, (120, 2))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        self.assertGreater(adjusted_rand_score(self.group, probability.argmax(axis=1)), 0.75)

    def test_subtype_prior_borrows_strength_for_new_patient(self) -> None:
        low = self.model.candidate_predict(
            self.features[0], cell="glucose@6h", base_point=0.0
        )
        high = self.model.candidate_predict(
            self.features[-1], cell="glucose@6h", base_point=0.0
        )
        self.assertLess(low.subtype_point, -1.0)
        self.assertGreater(high.subtype_point, 1.0)
        self.assertEqual(low.patient_weight, 0.0)
        self.assertEqual(low.promotion_status, "candidate_only")

    def test_resolved_personal_errors_gradually_override_subtype_prior(self) -> None:
        subtype_only = self.model.candidate_predict(
            self.features[-1], cell="glucose@6h", base_point=0.0
        )
        personalized = self.model.candidate_predict(
            self.features[-1],
            cell="glucose@6h",
            base_point=0.0,
            resolved_patient_errors=(4.0, 4.0, 4.0, 4.0),
        )
        self.assertAlmostEqual(personalized.patient_weight, 0.5)
        self.assertGreater(personalized.personalized_point, subtype_only.subtype_point)
        self.assertLess(personalized.personalized_point, 4.0)

    def test_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subtypes.joblib"
            self.model.save(path)
            loaded = SoftPhysiologySubtypeModel.load(path)
            before = self.model.responsibilities(self.features[:3])
            after = loaded.responsibilities(self.features[:3])
            np.testing.assert_allclose(before, after)

    def test_patient_level_placebo_breaks_residual_alignment(self) -> None:
        placebo = SoftPhysiologySubtypeModel(
            subtype_count=2,
            min_effective_subjects=10,
            subtype_prior_strength=1.0,
            patient_prior_strength=4.0,
            random_state=4,
        ).fit(
            self.features,
            base_points=self.base,
            observed_targets=self.targets,
            observed_mask=np.ones_like(self.targets, dtype=bool),
            subject_ids=self.subjects,
            cell_keys=("glucose@6h",),
            responsibility_permutation_seed=71,
        )
        candidate_correction = self.model.subtype_corrections(
            self.features, cell="glucose@6h"
        )
        placebo_correction = placebo.subtype_corrections(
            self.features, cell="glucose@6h"
        )
        self.assertGreater(np.std(candidate_correction), np.std(placebo_correction))
        self.assertGreater(
            np.corrcoef(candidate_correction, self.targets[:, 0])[0, 1],
            0.5,
        )

    def test_trajectory_aggregation_is_causal_and_patient_local(self) -> None:
        features = np.zeros((5, 311), dtype=np.float64)
        features[:, 0] = [1.0, 3.0, 100.0, 7.0, 9.0]
        subjects = np.asarray(["a", "a", "b", "a", "a"])
        times = np.asarray([0.0, 2.0, 2.0, 50.0, 51.0])
        transformed = causal_trajectory_features(
            features,
            subject_ids=subjects,
            time_hours=times,
            window_hours=48.0,
        )
        self.assertEqual(transformed[0, 0], 1.0)
        self.assertEqual(transformed[1, 0], 2.0)
        self.assertEqual(transformed[2, 0], 100.0)
        self.assertEqual(transformed[3, 0], 5.0)
        self.assertEqual(transformed[4, 0], 8.0)
        changed = features.copy()
        changed[4, 0] = 9999.0
        changed_transformed = causal_trajectory_features(
            changed,
            subject_ids=subjects,
            time_hours=times,
            window_hours=48.0,
        )
        self.assertEqual(changed_transformed[1, 0], transformed[1, 0])

    def test_hospital_balancing_equalizes_hospitals_and_patient_memberships(self) -> None:
        subjects = np.asarray(["a", "a", "b", "c", "c", "c", "a"])
        hospitals = np.asarray(["h1", "h1", "h1", "h2", "h2", "h2", "h2"])
        weights = _patient_equal_weights(subjects, hospitals)

        self.assertAlmostEqual(weights[hospitals == "h1"].sum(), 1.0)
        self.assertAlmostEqual(weights[hospitals == "h2"].sum(), 1.0)
        self.assertAlmostEqual(weights[[0, 1]].sum(), weights[[2]].sum())
        self.assertAlmostEqual(weights[[3, 4, 5]].sum(), weights[[6]].sum())

    def test_history_covariates_are_causal_and_patient_local(self) -> None:
        values = np.asarray([[1.0], [3.0], [100.0], [7.0], [9.0]])
        observed = np.ones_like(values, dtype=bool)
        ages = np.zeros_like(values)
        subjects = np.asarray(["a", "a", "b", "a", "a"])
        times = np.asarray([0.0, 2.0, 2.0, 50.0, 51.0])
        result = causal_history_covariates(
            values,
            observed_mask=observed,
            measurement_ages=ages,
            subject_ids=subjects,
            time_hours=times,
            window_hours=48.0,
        )
        self.assertEqual(result.shape, (5, 7))
        self.assertEqual(result[1, 0], 2.0)
        self.assertEqual(result[1, 1], 1.0)
        self.assertEqual(result[2, 0], 100.0)
        changed = values.copy()
        changed[4, 0] = 9999.0
        changed_result = causal_history_covariates(
            changed,
            observed_mask=observed,
            measurement_ages=ages,
            subject_ids=subjects,
            time_hours=times,
            window_hours=48.0,
        )
        np.testing.assert_allclose(changed_result[1], result[1])

    def test_target_specific_ridge_adapter_uses_continuous_phenotype(self) -> None:
        rng = np.random.default_rng(91)
        features = rng.normal(0.0, 0.05, size=(120, 311))
        phenotype = np.linspace(-2.0, 2.0, 120)[:, None]
        target = (1.5 * phenotype[:, 0])[:, None]
        model = SoftPhysiologySubtypeModel(
            subtype_count=2,
            min_effective_subjects=10,
            subtype_prior_strength=1.0,
            target_adapter="ridge",
            target_adapter_alpha=1.0,
            random_state=12,
        ).fit(
            features,
            base_points=np.zeros_like(target),
            observed_targets=target,
            observed_mask=np.ones_like(target, dtype=bool),
            subject_ids=np.asarray([f"r-{index}" for index in range(120)]),
            cell_keys=("map@6h",),
            phenotype_covariates=phenotype,
        )
        correction = model.subtype_corrections(
            features,
            cell="map@6h",
            phenotype_covariates=phenotype,
        )
        self.assertGreater(np.corrcoef(correction, target[:, 0])[0, 1], 0.9)

    def test_history_anchor_uses_complete_stay_before_sampling(self) -> None:
        values = np.asarray([[1.0], [3.0], [5.0], [100.0]])
        result = causal_history_covariates_at_anchors(
            values,
            observed_mask=np.ones_like(values, dtype=bool),
            measurement_ages=np.zeros_like(values),
            history_group_ids=np.asarray(["stay-a", "stay-a", "stay-a", "stay-b"]),
            history_times=np.asarray([0.0, 2.0, 4.0, 2.0]),
            anchor_group_ids=np.asarray(["stay-a", "stay-a"]),
            anchor_times=np.asarray([2.0, 4.0]),
            window_hours=6.0,
        )
        self.assertEqual(result.shape, (2, 7))
        self.assertEqual(result[0, 0], 2.0)
        self.assertEqual(result[1, 0], 3.0)
        self.assertEqual(result[1, 1], 1.0)

    def test_direct_ridge_reads_uncompressed_event_state(self) -> None:
        rng = np.random.default_rng(13)
        features = rng.normal(0.0, 0.05, size=(120, 311))
        phenotype = rng.normal(size=(120, 20))
        target = (2.0 * phenotype[:, 17])[:, None]
        model = SoftPhysiologySubtypeModel(
            subtype_count=2,
            min_effective_subjects=10,
            subtype_prior_strength=1.0,
            target_adapter="direct_ridge",
            target_adapter_alpha=1.0,
            random_state=4,
        ).fit(
            features,
            base_points=np.zeros_like(target),
            observed_targets=target,
            observed_mask=np.ones_like(target, dtype=bool),
            subject_ids=np.asarray([f"r-{index}" for index in range(120)]),
            cell_keys=("map@12h",),
            phenotype_covariates=phenotype,
        )
        correction = model.subtype_corrections(
            features,
            cell="map@12h",
            phenotype_covariates=phenotype,
        )
        self.assertGreater(np.corrcoef(correction, target[:, 0])[0, 1], 0.9)

    def test_clinical_events_are_unavailable_before_entry_time(self) -> None:
        frame = pd.DataFrame(
            {
                "stay_id": [7, 7],
                "onset_hour": [0.0, 0.0],
                "hours_since_onset": [1.0, 2.0],
            }
        )
        source = {
            category: {} for category in (
                "hemodynamic",
                "respiratory_support",
                "infection",
                "renal_support",
                "neurologic",
                "fluid_resuscitation",
            )
        }
        source["hemodynamic"][7] = np.asarray([30.0, 90.0])
        result = _clinical_event_matrix(frame, source)
        self.assertEqual(result.shape, (2, 24))
        self.assertAlmostEqual(result[0, 0], np.log1p(1))
        self.assertAlmostEqual(result[1, 0], np.log1p(1))
        self.assertAlmostEqual(result[1, 1], np.log1p(2))

    def test_unobserved_future_events_fail_closed(self) -> None:
        self.assertFalse(
            FUTURE_EVENT_POLICY["future_event_value_imputation_allowed"]
        )
        self.assertEqual(
            FUTURE_EVENT_POLICY["runtime_action_without_external_validation"],
            "validated_fallback_or_abstain",
        )
        self.assertTrue(
            FUTURE_EVENT_POLICY["clinical_event_context_is_observation_only"]
        )

    def test_treatment_covariates_reject_future_action_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "hist_vasopressor": [0, 1],
                "hist_vasopressor_evidence_count": [0, 3],
                "act_vasopressor": [1, 1],
            }
        )
        values = _treatment_covariates(
            frame,
            ("hist_vasopressor", "hist_vasopressor_evidence_count"),
        )
        np.testing.assert_allclose(values[:, 0], [0.0, 1.0])
        np.testing.assert_allclose(values[:, 1], [0.0, np.log(4.0)])
        with self.assertRaises(ValueError):
            _treatment_covariates(frame, ("act_vasopressor",))

    def test_group_robust_stack_uses_discovery_hospitals(self) -> None:
        truth = np.ones(20)
        base = np.zeros(20)
        phenotype = np.ones(20)
        row_level = np.full(20, 0.5)
        subjects = np.asarray([f"s-{index}" for index in range(20)])
        hospitals = np.asarray(["h1"] * 10 + ["h2"] * 10)
        contract = _select_group_robust_stack(
            truth=truth,
            base=base,
            phenotype=phenotype,
            row_level=row_level,
            subjects=subjects,
            hospitals=hospitals,
        )
        self.assertEqual(contract["phenotype_weight"], 1.0)
        self.assertEqual(contract["correction_scale"], 1.0)

    def test_consensus_gate_abstains_on_seed_disagreement(self) -> None:
        corrections = np.asarray(
            [
                [1.0, 1.0, -1.0],
                [1.2, -0.5, -1.1],
                [0.8, 0.6, -0.9],
            ]
        )
        result = _consensus_correction(corrections)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], -1.0)

    def test_selective_gate_requires_every_discovery_hospital_to_improve(self) -> None:
        truth = np.ones(40)
        base = np.zeros(40)
        candidate = np.ones(40)
        subjects = np.asarray([f"s-{index}" for index in range(40)])
        hospitals = np.asarray(["h1"] * 20 + ["h2"] * 20)
        contract = _choose_gate_threshold(
            truth=truth,
            base=base,
            candidate=candidate,
            probabilities=np.full(40, 0.9),
            subjects=subjects,
            hospitals=hospitals,
        )
        self.assertTrue(contract["enabled"])

        candidate[20:] = -1.0
        rejected = _choose_gate_threshold(
            truth=truth,
            base=base,
            candidate=candidate,
            probabilities=np.full(40, 0.9),
            subjects=subjects,
            hospitals=hospitals,
        )
        self.assertFalse(rejected["enabled"])

    def test_selective_gate_falls_back_when_disabled(self) -> None:
        base = np.asarray([1.0, 2.0])
        candidate = np.asarray([3.0, 4.0])
        covariates = np.zeros((2, 3))
        features = _selective_gate_features(
            base=base,
            candidate=candidate,
            phenotype_covariates=covariates,
        )
        self.assertEqual(features.shape, (2, 6))
        prediction, move_fraction = _apply_selective_gate(
            base=base,
            candidate=candidate,
            phenotype_covariates=covariates,
            contract={"enabled": False},
        )
        np.testing.assert_allclose(prediction, base)
        self.assertEqual(move_fraction, 0.0)

    def test_eicu_source_selection_prefers_hospital_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            narrow = pd.DataFrame(
                {
                    "stay_id": np.arange(20),
                    "hospitalid": np.repeat([1, 2], 10),
                    "map_t": 70.0,
                    "map_tp6": 72.0,
                }
            )
            broad = pd.DataFrame(
                {
                    "stay_id": np.arange(12),
                    "hospitalid": np.repeat([1, 2, 3, 4], 3),
                    "map_t": 70.0,
                    "map_tp6": 72.0,
                }
            )
            narrow.to_parquet(root / "eicu_narrow_transitions_6h.parquet")
            broad.to_parquet(root / "eicu_broad_transitions_6h.parquet")
            path, module = eicu_path(root, "map", 6)
            self.assertEqual(path, root / "eicu_broad_transitions_6h.parquet")
            self.assertEqual(module, "broad")

    def test_patient_update_uses_only_resolved_prior_errors(self) -> None:
        prediction, weights = _causal_patient_residual_update(
            base=np.zeros(3),
            candidate=np.ones(3),
            truth=np.asarray([4.0, 4.0, 4.0]),
            subjects=np.asarray(["p", "p", "p"]),
            anchor_times=np.asarray([0.0, 3.0, 6.0]),
            future_delta_hours=np.asarray([6.0, 6.0, 6.0]),
            prior_strength=4.0,
        )
        self.assertEqual(prediction[0], 1.0)
        self.assertEqual(prediction[1], 1.0)
        self.assertGreater(prediction[2], 1.0)
        self.assertEqual(weights[0], 0.0)
        self.assertEqual(weights[1], 0.0)
        self.assertGreater(weights[2], 0.0)


if __name__ == "__main__":
    unittest.main()
