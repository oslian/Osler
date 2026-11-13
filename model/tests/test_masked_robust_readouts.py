import unittest
from unittest.mock import patch

import numpy as np

from scripts.audit_masked_robust_readouts import choose_source_head
from scripts.masked_robust_readouts import estimator, fit_family, resampling, value_strata


class RobustReadoutTests(unittest.TestCase):
    def test_resampling_is_reproducible_and_donor_differs_before_bagging(self):
        subjects = np.asarray([f"p{i}" for i in range(50)])
        rows, permutation = resampling(subjects, "bun", 7)
        other, donor = resampling(subjects, "bun", 7)
        np.testing.assert_array_equal(rows, other)
        np.testing.assert_array_equal(permutation, donor)
        self.assertTrue((subjects[permutation[rows]] != subjects[rows]).all())

    def test_patient_overlap_rejected_before_fitting(self):
        pack = {"latent": np.ones((30, 2)), "y": np.ones(30),
                "subjects": np.asarray([f"p{i}" for i in range(30)]), "units": ["a"] * 30}
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_family("latent_median", pack, pack, "bun")

    def test_all_readouts_fit_and_preprocessing_is_fit_only(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(180, 3))
        y = x[:, 0] + rng.normal(size=180) * .1
        for family, parameters in (("latent_median", {"alpha": .01}),
                                   ("latent_huber", {"alpha": 1.}),
                                   ("latent_hist_mae", {"max_leaf_nodes": 7})):
            model = estimator(family, parameters, 7).fit(x, y)
            np.testing.assert_allclose(model[0].mean_, x.mean(axis=0))
            before = model[0].mean_.copy()
            self.assertTrue(np.isfinite(model.predict(x + 100)).all())
            np.testing.assert_array_equal(before, model[0].mean_)
        self.assertFalse(estimator("latent_hist_mae", {"max_leaf_nodes": 7}, 7)[-1].early_stopping)

    def test_unknown_family_rejected(self):
        with self.assertRaises(ValueError):
            estimator("not_a_method", {}, 7)

    def test_full_fit_keeps_equal_bag_count_and_matched_complexity(self):
        rng = np.random.default_rng(17)
        def pack(prefix):
            x = rng.normal(size=(100, 3))
            return {"latent": x, "y": x[:, 0] + .1 * rng.normal(size=100),
                    "subjects": np.asarray([f"{prefix}{i}" for i in range(100)]),
                    "units": np.asarray(["a"] * 50 + ["b"] * 50)}
        fit, tune = pack("fit"), pack("tune")
        with patch.dict("scripts.masked_robust_readouts.FAMILIES", {"latent_median": ({"alpha": .01},)}):
            bundle, info = fit_family("latent_median", fit, tune, "synthetic")
        self.assertEqual(info["status"], "fitted")
        self.assertEqual(info["selected"], 0)
        self.assertEqual(info["placebo_selected"], 0)
        for name in ("models", "placebos", "matched_placebos"):
            self.assertEqual(len(bundle[name]), 7)
            self.assertTrue(all(m[-1].alpha == .01 for m in bundle[name]))
        for seed_index, seed in enumerate((7, 19, 31, 43, 59, 71, 83)):
            rows, perm = resampling(fit["subjects"], "synthetic", seed)
            np.testing.assert_allclose(bundle["models"][seed_index][0].mean_, fit["latent"][rows].mean(axis=0))
            np.testing.assert_allclose(bundle["placebos"][seed_index][0].mean_, fit["latent"][perm[rows]].mean(axis=0))

    def test_failed_fitting_cannot_silently_select_a_model(self):
        def pack(prefix):
            return {"latent": np.ones((30, 2)), "y": np.ones(30),
                    "subjects": [f"{prefix}{i}" for i in range(30)], "units": ["a"] * 30}
        with patch("scripts.masked_robust_readouts.estimator", side_effect=ValueError("test failure")):
            bundle, info = fit_family("latent_median", pack("fit"), pack("tune"), "test")
        self.assertIsNone(bundle)
        self.assertEqual(info["status"], "all_configurations_failed")
        self.assertTrue(all(v["status"] == "fit_failed" for v in info["trace"].values()))

    def test_source_selector_only_reads_prespecified_tuning_scores(self):
        prior = {"latent_erm": {"chosen_steps": 128, "curve": {
                     "32": {"selection_score": .6}, "128": {"selection_score": .5}}},
                 "latent_ridge": {"alpha": 10., "curve": {"10.0": {"selection_score": .7}}}}
        info = {"latent_median": {"status": "fitted", "selection_score": .4},
                "latent_huber": {"status": "all_configurations_failed", "selection_score": 0.}}
        self.assertEqual(choose_source_head(info, prior)["chosen_arm"], "latent_median")

    def test_tail_statistics_keep_real_zero_and_do_not_modify_predictions(self):
        y = np.asarray([0., 2., 4., 10.])
        p = np.asarray([1., 2., 5., 7.])
        before = p.copy()
        result = value_strata(y, {"a": p}, [1., 6.])
        self.assertEqual(result["low"]["patients"], 1)
        self.assertEqual(result["high"]["models"]["a"]["bias"], -3.)
        np.testing.assert_array_equal(before, p)


if __name__ == "__main__":
    unittest.main()
