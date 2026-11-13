import unittest

import numpy as np
import pandas as pd

from scripts.train_observation_skip import NEW_ARMS, check_masked_target, fit_heads, inputs
from scripts.train_nonlinear_probe import PARAMETERS
from scripts.score_observation_skip import ARMS, losses, screen


class ObservationSkipTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.body = self.rng.normal(size=(60, 64))
        self.raw = self.rng.normal(size=(60, 265))

    def test_same_body_and_width_no_mutation(self):
        before = self.raw.copy()
        for arm in NEW_ARMS:
            x = inputs(self.body, self.raw, arm)
            self.assertEqual(x.shape, (60, 329))
            np.testing.assert_array_equal(x[:, :64], self.body)
        np.testing.assert_array_equal(self.raw, before)

    def test_process_branch_removes_only_numeric_summary(self):
        x = inputs(self.body, self.raw, 'process_skip')
        np.testing.assert_array_equal(x[:, 64:170], np.zeros((60, 106)))
        np.testing.assert_array_equal(x[:, 170:], self.raw[:, 106:])

    def test_pairing_only_changes_whole_extra_branch(self):
        donor = np.roll(np.arange(60), 1)
        x = inputs(self.body, self.raw, 'shuffled_skip', donor)
        np.testing.assert_array_equal(x[:, :64], self.body)
        np.testing.assert_array_equal(x[:, 64:], self.raw[donor])
        q = inputs(self.body, self.raw, 'shuffled_skip')
        np.testing.assert_array_equal(q[:, 64:], self.raw)
        with self.assertRaises(ValueError):
            inputs(self.body, self.raw, 'observation_skip', donor)
        with self.assertRaises(ValueError):
            inputs(self.body, self.raw, 'shuffled_skip', np.zeros(60, dtype=int))

    def test_target_mask_cannot_be_bypassed(self):
        with self.assertRaises(ValueError):
            check_masked_target(self.raw, 0)
        self.raw[:, [0, 53, 106, 212]] = 0
        check_masked_target(self.raw, 0)

    def test_wrong_width_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            inputs(self.body, self.raw[:, :-1], 'observation_skip')
        self.body[0, 0] = np.nan
        with self.assertRaises(ValueError):
            inputs(self.body, self.raw, 'observation_skip')

    def test_same_fit_budget_and_finite_predictions(self):
        models, unchanged = fit_heads(self.body, self.raw, self.raw[:, 0],
            [str(i) for i in range(60)], ['a']*59+['b'], 19)
        self.assertEqual(unchanged, 1)
        self.assertEqual(set(models), set(NEW_ARMS))
        for arm, model in models.items():
            self.assertEqual(model.n_features_in_, 329)
            self.assertEqual(model.n_iter_, 100)
            for k, v in PARAMETERS.items():
                self.assertEqual(model.get_params()[k], v)
            self.assertTrue(np.isfinite(model.predict(inputs(self.body, self.raw, arm))).all())

    def test_repeated_patient_averages_losses_not_predictions(self):
        frame = pd.DataFrame(dict(patient=['a', 'a', 'b'], fold=[0, 1, 0],
            target=['x']*3, component=['g1', 'g1', 'g2'], truth=[0., 0., 0.]))
        for arm in ARMS:
            frame[arm] = [-2., 2., 0.]
        people, groups = losses(frame)
        self.assertEqual(len(people), 2)
        self.assertEqual(people.loc['a', 'observation_skip'], 2.)
        self.assertEqual(groups.observation_skip.mean(), 1.)
        with self.assertRaises(ValueError):
            losses(pd.concat([frame, frame.iloc[:1]]))
        frame.loc[1, 'component'] = 'g2'
        with self.assertRaises(ValueError):
            losses(frame)

    def test_screen_requires_support_ci_and_corrected_p(self):
        good = dict(mean=-1., p_less=0., ci95=[-2., -.1])
        self.assertTrue(screen(dict(patient=good, component=good), 30, 3))
        self.assertFalse(screen(dict(patient=good, component=good), 29, 3))
        self.assertFalse(screen(dict(patient=good, component=good), 30, 2))
        for change in (dict(ci95=None), dict(ci95=[-2., .1]), dict(p_less=.01)):
            self.assertFalse(screen(dict(patient=good, component=dict(good, **change)), 30, 3))


if __name__ == '__main__':
    unittest.main()
