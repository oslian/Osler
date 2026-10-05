import unittest
import numpy as np
from scripts.audit_bun_conditional_residual import median_interval,summarize


class ConditionalTests(unittest.TestCase):
    def test_constant(self):
        result=median_interval(np.ones(30)*2)
        self.assertEqual((result['lower'],result['upper']),(2.,2.))

    def test_sparse_no_interval_and_boundary(self):
        rows=[dict(truth=4.,predictions={'full':2.})]
        bins=summarize(rows,np.array([2.]))
        self.assertEqual(bins[0]['patients'],0)
        self.assertIsNone(bins[1]['residual']['lower'])
        self.assertEqual(bins[1]['residual']['median'],2.)
