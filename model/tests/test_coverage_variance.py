import unittest
from scripts.diagnose_coverage_variance import decompose


class VarianceTests(unittest.TestCase):
    def test_known_noise(self):
        r=decompose([0,2],[.5,.5])
        self.assertEqual(r['observed_between_variance'],2)
        self.assertEqual(r['excess_variance'],1.5)

    def test_negative_excess_not_hidden(self):
        self.assertLess(decompose([0,1],[2,2])['excess_variance'],0)

    def test_unknown_not_zero(self):
        self.assertIsNone(decompose([0,1],[float('nan'),1])['estimated_sampling_variance'])


if __name__=='__main__':unittest.main()
