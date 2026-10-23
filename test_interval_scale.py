import unittest

import numpy as np

from osler_jepa.interval_scale import fit_interval_scale


def data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    current = rng.lognormal(5.0, 0.4, n)
    change = np.where(rng.random(n) < 0.3, np.nan, np.abs(rng.normal(0, 20, n)))
    volatile = np.nan_to_num(change, nan=0.0) > 30
    residual = np.abs(rng.normal(0, 1, n)) * current * 0.1 * np.where(volatile, 3.0, 1.0)
    subjects = np.arange(n) // 2
    return current, change, residual, subjects


class IntervalScaleTests(unittest.TestCase):
    def test_tail_bins_track_a_multiplicative_tail_better_than_quintiles(self):
        current, change, residual, subjects = data()
        top = current > np.quantile(current, 0.99)
        for scheme, tolerance in (("quintile", None), ("tail8", 0.25)):
            scale = fit_interval_scale(scheme, current, change, residual, subjects)
            ratio = np.median(residual[top] / scale(current[top], change[top]))
            if tolerance is not None:
                self.assertLess(abs(ratio - np.median(residual / scale(current, change))), tolerance)
        quintile = fit_interval_scale("quintile", current, change, residual, subjects)
        self.assertGreater(np.median(residual[top] / quintile(current[top], change[top])), 1.15)

    def test_change_factor_widens_volatile_rows_only_with_the_change_scheme(self):
        current, change, residual, subjects = data()
        volatile = np.nan_to_num(change, nan=0.0) > 30
        plain = fit_interval_scale("tail8", current, change, residual, subjects)
        scaled = fit_interval_scale("tail8_change", current, change, residual, subjects)
        ratio = scaled(current, change) / plain(current, change)
        self.assertGreater(np.median(ratio[volatile]), 1.5)
        self.assertLess(np.median(ratio[~volatile & np.isfinite(change)]), 1.0)

    def test_sparse_groups_fall_back_to_the_overall_scale(self):
        current, change, residual, subjects = data(n=100)
        scale = fit_interval_scale("tail8", current, change, residual, np.zeros(100))
        self.assertTrue(np.allclose(scale(current, change), np.median(residual)))


if __name__ == "__main__":
    unittest.main()
