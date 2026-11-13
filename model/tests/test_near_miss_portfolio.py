import unittest

import numpy as np

from whole_body_jepa.near_miss_portfolio import (
    bootstrap_simplex_weights,
    fit_expected_error_router,
    fit_simplex_weights,
    patient_equal_mae,
    predict_portfolio,
    route_expected_error,
    state_router_features,
    unique_prediction_columns,
)


class NearMissPortfolioTest(unittest.TestCase):
    def test_simplex_recovers_complementary_sources(self):
        rows = 60
        predictions = np.column_stack([np.zeros(rows), np.ones(rows)])
        truth = np.full(rows, 0.25)
        subjects = np.asarray([f"p{index // 2}" for index in range(rows)])
        weights = fit_simplex_weights(predictions, truth, subjects)
        np.testing.assert_allclose(weights, [0.75, 0.25], atol=1e-4)
        self.assertLess(
            patient_equal_mae(predict_portfolio(predictions, weights), truth, subjects),
            1e-5,
        )

    def test_bootstrap_weights_are_deterministic_and_on_simplex(self):
        rng = np.random.default_rng(7)
        predictions = rng.normal(size=(80, 3))
        truth = 0.2 * predictions[:, 0] + 0.8 * predictions[:, 2]
        subjects = np.asarray([f"p{index // 2}" for index in range(80)])
        first, bags_first = bootstrap_simplex_weights(
            predictions, truth, subjects, (7, 19, 31)
        )
        second, bags_second = bootstrap_simplex_weights(
            predictions, truth, subjects, (7, 19, 31)
        )
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(bags_first, bags_second)
        np.testing.assert_allclose(first.sum(), 1.0)
        self.assertTrue(np.all(first >= 0))

    def test_duplicate_sources_are_removed_without_truth(self):
        values = np.column_stack([np.arange(5), np.arange(5), np.arange(5) + 1])
        unique, names, aliases = unique_prediction_columns(
            values, ("persistence", "release", "history")
        )
        self.assertEqual(unique.shape, (5, 2))
        self.assertEqual(names, ("persistence", "history"))
        self.assertEqual(aliases, {"release": "persistence"})

    def test_state_router_can_learn_state_specific_source(self):
        rng = np.random.default_rng(11)
        rows = 600
        state = rng.normal(size=(rows, 2))
        truth = np.where(state[:, 0] < 0, -2.0, 2.0)
        predictions = np.column_stack([np.full(rows, -2.0), np.full(rows, 2.0)])
        current = np.zeros(rows)
        features = state_router_features(state, predictions, current, scale=1.0)
        subjects = np.asarray([f"p{index}" for index in range(rows)])
        heads = fit_expected_error_router(features, predictions, truth, subjects)
        routed, choice, expected = route_expected_error(heads, features, predictions)
        self.assertEqual(expected.shape, (rows, 2))
        self.assertGreater(np.mean(routed == truth), 0.95)
        self.assertGreater(np.mean(choice == (state[:, 0] >= 0)), 0.95)


if __name__ == "__main__":
    unittest.main()
