import unittest
import numpy as np
import pandas as pd
from scripts.try_conditional_reconstruction_repair import features, predict, source_cv


class RepairTests(unittest.TestCase):
    def frame(self):
        x = np.tile(np.arange(40.)/40, 4)
        return pd.DataFrame(dict(patient=[str(i) for i in range(160)], component=np.repeat(np.arange(4),40),
            real=1., process=1., population=1., truth=1+2*x, fit_scale=1.,
            observed_x=True, value_x=x, age_x=0.))

    def test_unseen_query_truth_cannot_affect_fit(self):
        source = self.frame(); query = source.copy()
        a, info = predict(source, query, ['x'])
        query['truth'] = 1e12
        b, _ = predict(source, query, ['x'])
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(info['input_variables'], ['x'])
        self.assertLess(np.mean(abs(a['numeric_context']-source.truth)), np.mean(abs(a['process_context']-source.truth)))

    def test_query_only_variable_not_used(self):
        source = self.frame(); query = source.copy()
        source['observed_x'] = False; source['value_x'] = np.nan; source['age_x'] = np.nan
        _, _, selected = features(source, query, ['x'])
        self.assertEqual(selected, [])

    def test_unsupported_exact_unchanged(self):
        source = self.frame().iloc[:40]; query = self.frame()
        prediction, info = predict(source, query, ['x'])
        self.assertEqual(info['status'], 'insufficient_support')
        for value in prediction.values():
            np.testing.assert_array_equal(value, query.real)

    def test_loco_held_component_truth_not_fitted(self):
        source = self.frame()
        before = source_cv(source, ['x'])
        source.loc[source.component == 0, 'truth'] += 10
        after = source_cv(source, ['x'])
        self.assertTrue(before['complete'])
        # Held component0 predictions remain centered at2 despite its new truth.
        self.assertAlmostEqual(after['components'][0]['mae']['process_context'], 10.)

    def test_squared_loss_can_harm_mae_before_transfer(self):
        source = self.frame()
        source['real'] = 0.; source['truth'] = 0.
        source.loc[[0,40,80,120], 'truth'] = 100.
        source['value_x'] = 0.
        p, _ = predict(source, source, ['x'])
        y = source.truth.to_numpy()
        self.assertLess(np.mean((p['numeric_context']-y)**2), np.mean(y*y))
        self.assertGreater(np.mean(abs(p['numeric_context']-y)), np.mean(abs(y)))


if __name__ == '__main__':
    unittest.main()
