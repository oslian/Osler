import unittest

import numpy as np

from whole_body_jepa.state_conditioned import (
    distribution_risk_samples,
    empirical_crps,
    expert_distribution_samples,
    fit_distribution_expert,
    fit_state_conditioned_head,
    fit_utility_conditioned_head,
    mixture_samples,
    predict_state_conditioned,
    predict_distribution_risk_conditioned,
    predict_utility_conditioned,
    utility_mixture_samples,
)


class StateConditionedTests(unittest.TestCase):
    def test_crps_is_zero_for_perfect_point_distribution(self) -> None:
        samples = np.asarray([[1.0] * 8, [2.0] * 8])
        np.testing.assert_allclose(empirical_crps(samples, np.asarray([1.0, 2.0])), 0.0)

    def test_mixture_uses_more_dynamic_samples_at_higher_probability(self) -> None:
        dynamic = np.asarray([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        samples = mixture_samples(
            np.asarray([0.0, 1.0]), dynamic, np.zeros(65), sample_count=16
        )
        np.testing.assert_allclose(samples[0], 0.0)
        self.assertGreater(float(samples[1].mean()), 2.0)

    def test_fit_separates_obvious_dynamic_state(self) -> None:
        rng = np.random.default_rng(7)
        shared = rng.normal(size=(800, 4))
        dynamic = shared[:, 0] > 0
        delta = np.where(dynamic, 2.0 + 0.1 * rng.normal(size=800), 0.05 * rng.normal(size=800))
        target = np.column_stack([shared, rng.normal(size=(800, 2))])
        head = fit_state_conditioned_head(
            shared, target, delta, dynamic_threshold=0.5, seed=7
        )
        prediction = predict_state_conditioned(head, shared, target, cutoff=0.5)
        self.assertGreater(
            float(prediction["dynamic_probability"][dynamic].mean()),
            float(prediction["dynamic_probability"][~dynamic].mean()),
        )
        self.assertEqual(prediction["point_delta"].shape, delta.shape)

    def test_utility_router_learns_when_expert_beats_persistence(self) -> None:
        rng = np.random.default_rng(19)
        expert_x = rng.normal(size=(1200, 4))
        expert_delta = 2.0 * expert_x[:, 0] + rng.normal(scale=0.1, size=1200)
        router_x = rng.normal(size=(1200, 4))
        router_delta = np.where(
            router_x[:, 0] > 0.0,
            2.0 * router_x[:, 0] + rng.normal(scale=0.1, size=1200),
            rng.normal(scale=0.03, size=1200),
        )
        head = fit_utility_conditioned_head(
            expert_x,
            expert_delta,
            router_x,
            router_x[:, :2],
            router_delta,
            seed=19,
        )
        prediction = predict_utility_conditioned(
            head, router_x, router_x[:, :2], probability_cutoff=0.6
        )
        useful = router_x[:, 0] > 0.5
        stable = router_x[:, 0] < -0.5
        self.assertGreater(float(prediction["move"][useful].mean()), 0.5)
        self.assertLess(float(prediction["move"][stable].mean()), 0.2)
        samples = utility_mixture_samples(prediction, head["persistence_delta_quantiles"])
        self.assertEqual(samples.shape, (1200, 64))

    def test_distribution_risk_moves_only_when_future_distribution_moves(self) -> None:
        far = expert_distribution_samples(np.asarray([[1.8, 2.0, 2.2]]), sample_count=32)
        near = expert_distribution_samples(np.asarray([[-0.2, 0.0, 0.2]]), sample_count=32)
        self.assertGreater(float(np.mean(np.abs(far))), 1.5)
        self.assertLess(float(np.mean(np.abs(near))), 0.2)

        rng = np.random.default_rng(23)
        features = rng.normal(size=(1000, 3))
        delta = 2.0 * features[:, 0] + rng.normal(scale=0.2, size=1000)
        head = fit_distribution_expert(features, delta, seed=23)
        prediction = predict_distribution_risk_conditioned(
            head, features, risk_margin=0.1, sample_count=32
        )
        self.assertGreater(float(prediction["move"][np.abs(features[:, 0]) > 1.0].mean()), 0.8)
        samples = distribution_risk_samples(prediction, head["persistence_delta_quantiles"])
        self.assertEqual(samples.shape, (1000, 32))


if __name__ == "__main__":
    unittest.main()
