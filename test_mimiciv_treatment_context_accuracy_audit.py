import unittest

import numpy as np
import pandas as pd

from mimiciv_treatment_context_accuracy_audit import (
    audit_split,
    conformal_comparison,
    evaluate_predictions,
    feature_sets,
    fit_ridge_residual,
)


class MimicIVTreatmentContextAccuracyAuditTests(unittest.TestCase):
    def test_feature_sets_separate_treatment_context_from_baseline(self):
        frame = pd.DataFrame({
            "subject_id": [1, 2],
            "stay_id": [10, 20],
            "first_careunit": ["MICU", "SICU"],
            "glucose_t": [100.0, 120.0],
            "glucose_age_hr": [0.1, 0.2],
            "glucose_tp6": [130.0, 150.0],
            "map_t": [70.0, 72.0],
            "map_age_hr": [0.1, 0.2],
            "hist_insulin": [0, 1],
            "hist_insulin_dose_drug_units_sum": [0.0, 5.0],
            "hist_antibiotics": [1, 0],
            "hist_antibiotics_dose_mass_mg_sum": [500.0, 0.0],
            "act_insulin": [1, 0],
            "act_insulin_evidence_count": [2, 0],
            "active_sepsis": [False, False],
        })

        sets = feature_sets(frame, "glucose")

        self.assertIn("map_t", sets["baseline"])
        self.assertNotIn("hist_insulin", sets["baseline"])
        self.assertNotIn("act_insulin", sets["baseline"])
        self.assertIn("hist_insulin", sets["candidate"])
        self.assertIn("hist_insulin_dose_drug_units_sum", sets["candidate"])
        self.assertNotIn("hist_antibiotics", sets["candidate"])
        self.assertNotIn("hist_antibiotics_dose_mass_mg_sum", sets["candidate"])
        self.assertNotIn("act_insulin", sets["candidate"])
        self.assertNotIn("act_insulin_evidence_count", sets["treatment"])

    def test_feature_sets_are_target_specific_for_hematocrit(self):
        frame = pd.DataFrame({
            "subject_id": [1, 2],
            "stay_id": [10, 20],
            "first_careunit": ["MICU", "SICU"],
            "hematocrit_t": [30.0, 31.0],
            "hematocrit_age_hr": [0.1, 0.2],
            "hematocrit_tp6": [29.0, 30.0],
            "hist_fluids": [1, 0],
            "hist_fluids_dose_volume_ml_sum": [500.0, 0.0],
            "hist_transfusion": [0, 1],
            "hist_transfusion_dose_volume_ml_sum": [0.0, 250.0],
            "hist_insulin": [1, 1],
            "hist_insulin_dose_drug_units_sum": [4.0, 5.0],
        })

        sets = feature_sets(frame, "hematocrit")

        self.assertIn("hist_fluids", sets["treatment"])
        self.assertIn("hist_fluids_dose_volume_ml_sum", sets["treatment"])
        self.assertIn("hist_transfusion", sets["treatment"])
        self.assertNotIn("hist_insulin", sets["candidate"])
        self.assertNotIn("hist_insulin_dose_drug_units_sum", sets["candidate"])

    def test_fit_ridge_residual_supports_equal_capacity_noise(self):
        rows = []
        for subject in range(30):
            for idx in range(2):
                current = 100.0 + subject * 0.5 + idx
                treatment = float(subject % 2)
                rows.append({
                    "subject_id": subject,
                    "glucose_t": current,
                    "glucose_tp6": current + 5.0 * treatment,
                    "map_t": 70.0 + idx,
                    "hist_insulin": treatment,
                    "truth": current + 5.0 * treatment,
                    "current": current,
                })
        frame = pd.DataFrame(rows)
        train = frame.iloc[:40].copy()
        predict = frame.iloc[40:].copy()

        baseline = fit_ridge_residual(
            train,
            predict,
            "glucose",
            ["map_t"],
            alpha=1.0,
            seed=1,
        )
        placebo = fit_ridge_residual(
            train,
            predict,
            "glucose",
            ["map_t"],
            alpha=1.0,
            seed=1,
            noise_count=1,
        )
        candidate = fit_ridge_residual(
            train,
            predict,
            "glucose",
            ["map_t", "hist_insulin"],
            alpha=1.0,
            seed=1,
        )

        self.assertEqual(len(baseline), len(predict))
        self.assertEqual(len(placebo), len(predict))
        self.assertLess(
            np.abs(candidate - predict["truth"].to_numpy(dtype=float)).mean(),
            np.abs(baseline - predict["truth"].to_numpy(dtype=float)).mean(),
        )

    def test_audit_split_requires_candidate_to_beat_baseline_placebo_and_persistence(self):
        rows = []
        for subject in range(80):
            group = "train" if subject < 50 else "test"
            treatment = float(subject % 2)
            for idx in range(2):
                current = 100.0 + idx
                future = current + 10.0 * treatment
                rows.append({
                    "subject_id": subject,
                    "split_group": group,
                    "glucose_t": current,
                    "glucose_tp6": future,
                    "map_t": 70.0 + idx,
                    "hist_insulin": treatment,
                    "truth": future,
                    "current": current,
                })
        frame = pd.DataFrame(rows)
        report = audit_split(
            frame,
            "glucose",
            ["map_t"],
            ["map_t", "hist_insulin"],
            1,
            {"train"},
            {"test"},
            "split_group",
            seed=7,
            alpha=1.0,
            inner_folds=3,
            bootstrap_samples=100,
            min_pairs=20,
            min_subjects=10,
        )

        self.assertEqual(report["status"], "evaluated")
        self.assertTrue(report["selected_candidate"])
        self.assertTrue(report["heldout"]["candidate_vs_baseline"]["significant"])
        self.assertTrue(report["heldout"]["candidate_vs_placebo"]["significant"])
        self.assertTrue(report["heldout"]["candidate_vs_persistence"]["significant"])

    def test_candidate_equal_to_persistence_cannot_pass(self):
        rows = pd.DataFrame({
            "subject_id": np.repeat(np.arange(40), 2),
            "truth": np.tile([100.0, 101.0], 40),
            "current": np.tile([99.0, 100.0], 40),
        })
        persistence = rows["current"].to_numpy(dtype=float)
        result = evaluate_predictions(
            rows,
            baseline=np.full(len(rows), 80.0),
            candidate=persistence,
            placebo=np.full(len(rows), 120.0),
            seed=11,
            bootstrap_samples=100,
        )

        self.assertFalse(result["candidate_vs_persistence"]["significant"])

    def test_conformal_comparison_uses_patient_equalized_residuals(self):
        discovery = pd.DataFrame({
            "subject_id": ["a", "a", "b", "b"] * 10,
            "truth": np.tile([0.0, 2.0, 0.0, 2.0], 10),
        })
        heldout = pd.DataFrame({
            "subject_id": ["c", "c", "d", "d"],
            "truth": [0.0, 2.0, 0.0, 2.0],
        })
        baseline_discovery = np.ones(len(discovery))
        candidate_discovery = discovery["truth"].to_numpy(dtype=float) + 0.25
        report = conformal_comparison(
            discovery,
            heldout,
            {"baseline": baseline_discovery, "candidate": candidate_discovery},
            {
                "baseline": np.ones(len(heldout)),
                "candidate": heldout["truth"].to_numpy(dtype=float) + 0.25,
            },
        )

        self.assertTrue(report["candidate_narrower"])
        self.assertLess(
            report["candidate"]["full_width"],
            report["baseline"]["full_width"],
        )

    def test_domain_adaptive_conformal_never_uses_heldout_labels_for_selection(self):
        discovery = pd.DataFrame({
            "subject_id": [f"p{index // 2}" for index in range(80)],
            "first_careunit": ["MICU"] * 20 + ["SICU"] * 20 + ["CCU"] * 20 + ["CVICU"] * 20,
            "truth": np.r_[np.ones(20), np.ones(20) * 2, np.ones(20) * 3, np.ones(20) * 4],
        })
        heldout = pd.DataFrame({
            "subject_id": [f"h{index // 2}" for index in range(20)],
            "first_careunit": ["TSICU"] * 20,
            "truth": np.ones(20) * 100.0,
        })
        discovery_point = np.zeros(len(discovery))
        heldout_point = np.zeros(len(heldout))

        report = conformal_comparison(
            discovery,
            heldout,
            {"baseline": discovery_point, "candidate": discovery_point},
            {"baseline": heldout_point, "candidate": heldout_point},
            group_column="first_careunit",
        )

        self.assertFalse(report["baseline"]["selection_uses_heldout_labels"])
        self.assertIsNotNone(report["baseline"]["calibration_cross_domain_coverage"])


if __name__ == "__main__":
    unittest.main()
