import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import adaptive_cell_weights as weights
from mimic_to_eicu_49_cell_external_audit import fit_continuous_recipe, normalize_cohort


def transitions(n_stays, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for stay in range(n_stays):
        level = rng.normal(4.0, 0.5)
        for k in range(10):
            value = level + rng.normal(0, 0.3)
            rows.append({"subject_id": stay, "stay_id": stay, "first_careunit": ["MICU", "SICU"][stay % 2],
                         "t": pd.Timestamp("2100-01-01") + pd.Timedelta(hours=6 + 6 * k),
                         "potassium_t": value, "potassium_age_hr": 1.0, "sodium_t": 140 + rng.normal(0, 3), "sodium_age_hr": 2.0,
                         "potassium_tp6": 0.6 * value + 1.6 + rng.normal(0, 0.3) if rng.random() < 0.9 else np.nan,
                         "potassium_tp6_offset_hr": rng.uniform(-2, 2)})
    return pd.DataFrame(rows)


class AdaptiveCellWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame, _ = normalize_cohort(transitions(400, 0), database="MIMIC-IV", target="potassium", horizon=6, max_rows=0)
        cls.frame = frame.sort_values(["subject_id", "anchor_time"]).reset_index(drop=True)
        cls.train = np.arange(len(cls.frame))

    def reproduce(self, recipe):
        fitted = fit_continuous_recipe(self.frame, self.train, self.train, target="potassium", source=recipe, seed=7)
        bundle = weights.fit_predictor(self.frame, self.train, target="potassium", recipe=recipe, seed=7,
                                       policy=fitted.get("mimic_selected_policy"))
        with tempfile.TemporaryDirectory() as folder:
            joblib.dump(bundle, Path(folder) / "b.joblib")
            loaded = joblib.load(Path(folder) / "b.joblib")
        rows = fitted["calibration_rows"]
        self.assertTrue(weights.reproduces(weights.predict(loaded, self.frame)[rows], fitted["calibration_prediction"]))
        return fitted

    def test_ridge_bundle_reproduces_the_frozen_fit(self):
        self.reproduce("measurement_pure_ridge")

    def test_hurdle_bundle_reproduces_the_frozen_policy(self):
        self.assertTrue(self.reproduce("target_specific_change_hurdle")["mimic_selected_policy"])

    def test_persistence_bundle_returns_the_current_value(self):
        self.reproduce("persistence")

    def test_reproduction_tolerance_is_float32_relative_to_the_largest_value(self):
        frozen = np.array([3000.0, 5.0])
        self.assertTrue(weights.reproduces(frozen + [0.0, 2.0 ** -12], frozen))      # one float32 step at 2048-4096
        self.assertFalse(weights.reproduces(frozen + [0.0, 0.01], frozen))

    def test_every_hurdle_policy_parses(self):
        for name, kind in (("robust_compact_delta_0.25", "robust"), ("ridge_0.75_robust_interaction_0.25", "blend"),
                           ("soft_p_x_delta_1", "soft"), ("hard_p0.35_x_delta_0.5", "hard")):
            self.assertEqual(weights.parse_policy(name)[0], kind)
        with self.assertRaises(ValueError):
            weights.parse_policy("unknown")

    def test_interval_start_matches_the_scorer(self):
        fitted = fit_continuous_recipe(self.frame, self.train, self.train, target="potassium", source="measurement_pure_ridge", seed=7)
        start = weights.interval_start(self.frame, fitted, "potassium")
        self.assertGreater(start["point_scale"], 0)
        self.assertTrue(np.all(np.isfinite(start["calibration_scores"])) and len(start["calibration_scores"]) > 0)


if __name__ == "__main__":
    unittest.main()
