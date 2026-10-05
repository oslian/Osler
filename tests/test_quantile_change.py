import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.audit_prospective_change import evaluate, evaluate_predictions
from whole_body_jepa.quantile_change import (
    QUANTILES, tail_probabilities, pinball_score, fit_pooled_cqr,
    pooled_bounds, paired_ranking_recall,
)


class QuantileChangeTests(unittest.TestCase):
    def test_distribution_probabilities_include_threshold_atoms(self):
        q = np.stack([np.zeros(19), np.full(19, 10.), np.full(19, -10.)])
        np.testing.assert_allclose(tail_probabilities(q, 10), [[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    def test_linear_quantile_cdf_and_monotone_thresholds(self):
        q = (QUANTILES * 20 - 10)[None, :]
        p = tail_probabilities(q, 5)
        np.testing.assert_allclose(p, [[.25, .5, .25]])
        self.assertGreater(tail_probabilities(q, 6)[0, 1], p[0, 1])
        with self.assertRaises(ValueError):
            tail_probabilities(q[:, ::-1], 5)
        with self.assertRaises(ValueError):
            tail_probabilities(q, 0)

    def test_pinball_score_not_point_mae(self):
        q = np.zeros((2, 19))
        np.testing.assert_allclose(pinball_score([0, 2], q), [0, 1])

    def test_calibration_uses_distinct_patients_and_preserves_conditional_width(self):
        q = np.tile([-1., 0., 1.], (100, 1))
        truth = np.full(100, 3.)
        subjects = [str(i) for i in range(100)]
        r = fit_pooled_cqr(truth, q, subjects, 1., normalized=False)
        self.assertEqual(r, 2.)
        normalized = fit_pooled_cqr(truth, q, subjects, 1., normalized=True)
        self.assertEqual(normalized, 1.)
        test_q = np.array([[-1., 0., 1.], [-2., 0., 2.]])
        lo, hi = pooled_bounds(test_q, 1., normalized, True)
        np.testing.assert_allclose(hi - lo, [6, 12])
        with self.assertRaises(ValueError):
            fit_pooled_cqr(truth, q, ["same"] * 100, 1., False)
        self.assertIsNone(fit_pooled_cqr(truth[:99], q[:99], subjects[:99], 1., False))
        self.assertTrue(np.isnan(pooled_bounds(test_q, 1., None, False)[0]).all())

    def test_paired_ranking_identical_scores_have_zero_difference(self):
        labels = np.arange(100) >= 50
        score = np.arange(100) / 100
        result = paired_ranking_recall(labels, score, score, repetitions=20)
        self.assertEqual(result["mean"], 0.)
        self.assertEqual(result["ci95"], [0., 0.])

    def test_supplied_prediction_evaluator_matches_previous_entrypoint(self):
        n = 60
        f = pd.DataFrame({"subject_id": [str(i) for i in range(n)], "stay_id": np.arange(n),
            "t": pd.to_datetime(["2020-01-01"] * n), "_time_hours": np.ones(n),
            "future_event_time_map": pd.to_datetime(["2020-01-01 03:00"] * n),
            "map_t": np.full(n, 80.), "future_map": np.where(np.arange(n) % 2, 100., 80.),
            "careunit": ["a"] * 30 + ["b"] * 30})
        pred = {"point": np.full(n, 85.), "lower": np.full(n, 60.), "upper": np.full(n, 105.),
                "risk": np.full(n, .5), "alert": np.zeros(n, dtype=bool)}
        predictions = {"direct": pred, "persistence": {**pred, "point": np.full(n, 80.)}}
        cutoffs = {"direct": .6, "persistence": .6}
        high = np.arange(n) % 3 == 0
        direct = evaluate_predictions({"frame": f}, predictions, high, cutoffs, "map", 10)
        with patch("scripts.audit_prospective_change.calibrated_predictions", return_value=(predictions, high)):
            wrapped = evaluate({"frame": f}, {"cutoffs": cutoffs}, "map", 10)
        self.assertEqual(direct, wrapped)


if __name__ == "__main__":
    unittest.main()
