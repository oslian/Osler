import unittest

import numpy as np
import pandas as pd

from whole_body_jepa.expected_error_routing import (
    expected_advantage, fit_expected_error_router, route_delta,
    forecast_evidence, fit_cost_router, cost_advantage, FORECAST_EVIDENCE_NAMES,
)


class ExpectedErrorTests(unittest.TestCase):
    def test_forecast_routing_audit_runs_end_to_end(self):
        from scripts.audit_forecast_aware_routing import NEW_METHODS, calibrate, evaluate
        from whole_body_jepa.state_conditioned import expert_distribution_samples

        rng = np.random.default_rng(71)
        n = 400
        proposal = rng.normal(0, .1, n)
        delta = proposal + rng.normal(0, .2, n)
        pred = {"expert": proposal,
                "expert_samples": expert_distribution_samples(np.column_stack((proposal - .4, proposal, proposal + .4))),
                "persistence_samples": np.linspace(-.5, .5, 64),
                "q25": np.full(n, -1.),
                "scores": {"mean_benefit": rng.normal(size=n), "old_q25": np.ones(n)},
                "new_scores": {m: rng.normal(size=n) for m in NEW_METHODS}}
        old = {"primary": "mean_benefit", "thresholds": {"mean_benefit": 0, "old_q25": .5}}
        thresholds = {m: 0 for m in NEW_METHODS}
        scales = calibrate(pred, delta, thresholds, old)
        frame = pd.DataFrame({"subject_id": [str(i) for i in range(n)],
                              "careunit": np.arange(n) % 4, "_time_hours": np.arange(n),
                              "creatinine_t": 1., "future_creatinine": 1 + delta})
        result = evaluate(frame, pred, thresholds, scales, old, "creatinine", "synthetic", "aware_mean")
        self.assertEqual(result["methods"]["old_q25"]["move_fraction"], 0)
        self.assertAlmostEqual(result["methods"]["persistence"]["mae"], np.abs(delta).mean())
        self.assertAlmostEqual(result["methods"]["always_expert"]["mae"], np.abs(delta - proposal).mean())
        self.assertIn("shuffled_mean", result["methods"]["aware_mean"]["comparisons"])

    def test_forecast_evidence_preserves_zero_and_exposes_proposal(self):
        x = forecast_evidence([0, 2], [0, 1], [1, -1], [[2, 1, 0], [-2, -1, 0]], scale=2)
        self.assertEqual(x.shape, (2, len(FORECAST_EVIDENCE_NAMES)))
        self.assertEqual(x[0, 0], 0)
        np.testing.assert_allclose(x[:, 2], [.5, -.5])
        np.testing.assert_allclose(x[:, -1], [1, 1])
        with self.assertRaises(ValueError):
            forecast_evidence([0], [-1], [1], [[0, 1, 2]], scale=1)

    def test_cost_router_weights_harm_not_win_count(self):
        x = np.zeros((1000, 1))
        truth = np.ones(1000)
        proposal = np.ones(1000)
        proposal[::10] = 21
        model = fit_cost_router(x, truth, proposal)
        self.assertTrue(np.all(cost_advantage(model, x[:5]) < 0))

    def test_expert_output_distinguishes_states_hidden_from_old_router(self):
        # Identical latent, but one expert proposes the correct delta and another
        # overshoots. The gate can distinguish them only after seeing the proposal.
        proposal = np.repeat([1., 5.], 500)
        truth = np.ones(1000)
        q = np.column_stack((proposal - .5, proposal, proposal + .5))
        x = forecast_evidence(np.ones(1000), np.zeros(1000), proposal, q, scale=1)
        gate = fit_cost_router(x, truth, proposal)
        score = cost_advantage(gate, x[[0, -1]])
        self.assertGreater(score[0], 0)
        self.assertLess(score[1], 0)

    def test_zero_regret_does_not_invent_a_preference(self):
        x = np.zeros((100, 1))
        model = fit_cost_router(x, np.ones(100), np.zeros(100))
        np.testing.assert_array_equal(cost_advantage(model, x), np.zeros(100))

    def test_cross_hospital_patients_never_enter_development(self):
        from scripts.audit_state_conditioned_kidney import source_roles
        from scripts.audit_expected_error_routing import assert_disjoint, calibrating_split
        from whole_body_jepa.multisource_intervals import stable_hash

        held = next(str(h) for h in range(100) if stable_hash("state_conditioned_hospital", str(h)) % 5 == 0)
        dev = next(str(h) for h in range(100) if stable_hash("state_conditioned_hospital", str(h)) % 5 != 0)
        frame = pd.DataFrame({"subject_id": ["cross", "cross"] + [f"p{i}" for i in range(100)],
                              "hospitalid": [held, dev] + [dev] * 100})
        roles = {name: np.flatnonzero(mask) for name, mask in source_roles(frame).items()}
        selection, calibration = calibrating_split(frame, roles.pop("tune"))
        roles.update(selection=selection, calibration=calibration)
        assert_disjoint(frame, roles)
        for name, rows in roles.items():
            if name != "hospital":
                self.assertNotIn("cross", set(frame.iloc[rows].subject_id))
        np.testing.assert_array_equal(roles["hospital"], [0])

    def test_mean_loss_handles_infrequent_large_harm(self):
        # Expert wins 90% of cases, but rare large errors make its mean MAE worse.
        x = np.zeros((1000, 2))
        truth = np.ones(1000)
        prediction = np.ones(1000)
        prediction[::10] = 21.0
        model = fit_expected_error_router(x, truth, prediction)
        for direct in (False, True):
            advantage = expected_advantage(model, x[:5], direct=direct)
            self.assertTrue(np.all(advantage < 0))
            _, move = route_delta(np.ones(5), advantage)
            self.assertFalse(move.any())

    def test_identifiable_regimes_choose_opposite_experts(self):
        x = np.repeat([[-1.0], [1.0]], 500, axis=0)
        truth = np.where(x[:, 0] > 0, 2.0, 0.0)
        model = fit_expected_error_router(x, truth, np.full(1000, 2.0))
        advantage = expected_advantage(model, np.array([[-1.0], [1.0]]))
        point, move = route_delta(np.full(2, 2.0), advantage)
        np.testing.assert_array_equal(move, [False, True])
        np.testing.assert_array_equal(point, [0.0, 2.0])

    def test_invalid_training_fails(self):
        with self.assertRaises(ValueError):
            fit_expected_error_router(np.zeros((3, 2)), np.ones(2), np.ones(3))
        with self.assertRaises(ValueError):
            fit_expected_error_router(np.full((3, 2), np.nan), np.ones(3), np.ones(3))
