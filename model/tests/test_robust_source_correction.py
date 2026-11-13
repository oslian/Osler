import unittest
import numpy as np
from tests.test_conditional_reconstruction_repair import RepairTests
from scripts.try_robust_source_correction import robust_predict


class RobustTests(unittest.TestCase):
    def test_query_truth_independent(self):
        source = RepairTests().frame()
        source['truth'] += np.random.default_rng(19).normal(0,.1,len(source))
        query = source.copy()
        a, info = robust_predict(source, query, ['x'])
        self.assertEqual(info['status'], 'fitted')
        query['truth'] = 1e9
        b, _ = robust_predict(source, query, ['x'])
        for arm in a:
            np.testing.assert_array_equal(a[arm], b[arm])

    def test_insufficient_not_successful_zero(self):
        source = RepairTests().frame().iloc[:40]
        p, info = robust_predict(source, source, ['x'])
        self.assertEqual(p, {})
        self.assertEqual(info['status'], 'insufficient_support')


if __name__ == '__main__':
    unittest.main()
