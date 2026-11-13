import unittest

import numpy as np

from osler_jepa.site_adaptive_conformal import (
    AdaptiveConfig,
    DtaciConfig,
    adaptive_stream_widths,
    dtaci_stream_widths,
    covariate_shift_weights,
    weighted_quantile,
)


def stream(n_stays=400, rows=10, scale=1.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_stays * rows
    stays = np.repeat(np.arange(n_stays), rows)
    anchors = np.arange(n) * 6.0          # one shared clock: stays follow each other
    return {
        "stays": stays,
        "levels": {"patient": stays.astype(str)},
        "anchor_hours": anchors,
        "resolve_hours": anchors + 6.0,
        "scales": np.full(n, scale),
        "predictions": np.zeros(n),
        "observed": rng.normal(scale=noise, size=n),
    }


class SiteAdaptiveConformalTests(unittest.TestCase):
    def setUp(self):
        self.scores = np.abs(np.random.default_rng(1).normal(size=5000))
        self.weights = np.ones_like(self.scores)

    def test_weighted_quantile_matches_unweighted_with_equal_weights(self):
        q = weighted_quantile(self.scores, self.weights, 0.9)
        self.assertAlmostEqual(q, np.quantile(self.scores, 0.9), delta=0.02)

    def test_without_adaptation_a_noisier_site_is_undercovered(self):
        data = stream(noise=2.0)
        result = adaptive_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                        config=AdaptiveConfig(etas={}), **data)
        coverage = np.mean(np.abs(data["observed"]) <= result.widths)
        self.assertLess(coverage, 0.65)

    def test_site_adaptation_recovers_nominal_coverage_under_shift(self):
        data = stream(noise=2.0)
        result = adaptive_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                        config=AdaptiveConfig(etas={"site": 0.05}), **data)
        covered = np.abs(data["observed"]) <= result.widths
        self.assertAlmostEqual(float(np.mean(covered[1000:])), 0.90, delta=0.03)

    def test_patient_term_adapts_to_a_noisier_patient(self):
        data = stream(n_stays=1, rows=400)
        data["observed"] = np.random.default_rng(3).normal(scale=3.0, size=400)
        data["resolve_hours"] = data["anchor_hours"] + 1.0
        site_only = adaptive_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                           config=AdaptiveConfig(etas={}), **data)
        with_patient = adaptive_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                              config=AdaptiveConfig(etas={"patient": 0.05}), **data)
        late = slice(200, None)
        covered = lambda result: float(np.mean(np.abs(data["observed"][late]) <= result.widths[late]))
        self.assertGreater(covered(with_patient), covered(site_only) + 0.15)

    def test_unit_level_fixes_one_undercovered_unit_that_site_level_cannot(self):
        data = stream(n_stays=600, rows=10)
        units = np.where(data["stays"] % 5 == 0, "bad", "good")
        noise = np.where(units == "bad", 3.0, 1.0)
        data["observed"] = np.random.default_rng(4).normal(size=len(units)) * noise
        data["levels"] = {"care_unit": units}
        late = np.arange(len(units)) >= 2000
        bad = (units == "bad") & late
        site = adaptive_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                      config=AdaptiveConfig(etas={"site": 0.05}), **data)
        unit = adaptive_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                      config=AdaptiveConfig(etas={"site": 0.05, "care_unit": 0.05}), **data)
        coverage = lambda result: float(np.mean(np.abs(data["observed"][bad]) <= result.widths[bad]))
        self.assertLess(coverage(site), 0.8)
        self.assertAlmostEqual(coverage(unit), 0.90, delta=0.04)

    def test_dtaci_adapts_faster_than_a_slow_fixed_step_on_a_short_stream(self):
        data = stream(n_stays=40, rows=8, noise=2.5)
        data["levels"] = {"care_unit": np.zeros(len(data["stays"]), dtype=int)}
        slow = adaptive_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                      config=AdaptiveConfig(etas={"site": 0.005}), **data)
        dtaci = dtaci_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                    config=DtaciConfig(), **data)
        coverage = lambda result: float(np.mean(np.abs(data["observed"]) <= result.widths))
        self.assertGreater(coverage(dtaci), coverage(slow) + 0.05)

    def test_dtaci_weights_survive_an_extreme_outlier(self):
        data = stream(n_stays=50, rows=10, noise=1.0)
        data["observed"][5] = 1.0e9
        data["levels"] = {"care_unit": np.zeros(len(data["stays"]), dtype=int)}
        result = dtaci_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                     config=DtaciConfig(meta_rate=20.0), **data)
        self.assertTrue(np.all(np.isfinite(result.widths)))

    def test_dtaci_stays_near_nominal_without_shift(self):
        data = stream(n_stays=400, rows=10, noise=1.0)
        data["levels"] = {"care_unit": np.zeros(len(data["stays"]), dtype=int)}
        result = dtaci_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                     config=DtaciConfig(), **data)
        self.assertAlmostEqual(float(np.mean(np.abs(data["observed"]) <= result.widths)), 0.90, delta=0.03)

    def test_dtaci_initial_multiplier_widens_the_first_interval_and_still_adapts(self):
        data = stream(n_stays=400, rows=10, noise=1.0)
        data["levels"] = {"care_unit": np.zeros(len(data["stays"]), dtype=int)}
        plain = dtaci_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                    config=DtaciConfig(), **data)
        prior = dtaci_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                    config=DtaciConfig(initial_log_multiplier=0.3), **data)
        self.assertAlmostEqual(prior.widths[0] / plain.widths[0], float(np.exp(0.3)), places=6)
        self.assertAlmostEqual(float(np.mean(np.abs(data["observed"]) <= prior.widths)), 0.90, delta=0.03)

    def test_dtaci_stays_bounded_and_calibrated_under_long_feedback_delay(self):
        # A network-like stream: ~1,000 outcomes pending at any time and a site
        # noisier than the calibration data, so early thresholds are too narrow.
        data = stream(n_stays=1500, rows=8, noise=2.0)
        n = len(data["stays"])
        data["anchor_hours"] = np.arange(n) * 0.05
        data["resolve_hours"] = data["anchor_hours"] + 50.0
        data["levels"] = {"care_unit": np.zeros(n, dtype=int)}
        result = dtaci_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                     config=DtaciConfig(meta_rate=20.0, initial_log_multiplier=0.15), **data)
        self.assertLess(float(np.max(result.log_multiplier)), 3.0)                     # no runaway multiplier
        late = slice(n // 2, None)
        self.assertAlmostEqual(float(np.mean(np.abs(data["observed"][late]) <= result.widths[late])), 0.90, delta=0.03)

    def test_outcomes_are_never_used_before_they_resolve(self):
        data = stream(n_stays=1, rows=5)
        data["observed"] = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        data["resolve_hours"] = data["anchor_hours"] + 1000.0  # never resolves inside the stay
        result = adaptive_stream_widths(calibration_scores=self.scores, calibration_weights=self.weights,
                                        config=AdaptiveConfig(etas={"site": 0.5}), **data)
        self.assertTrue(np.allclose(result.widths, result.widths[0]))

    def test_covariate_shift_weights_upweight_target_like_rows(self):
        rng = np.random.default_rng(2)
        source = rng.normal(size=(4000, 1))
        target = rng.normal(loc=1.0, size=(4000, 1))
        weights = covariate_shift_weights(source, target)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(weights[source[:, 0] > 1].mean(), weights[source[:, 0] < -1].mean())


if __name__ == "__main__":
    unittest.main()
