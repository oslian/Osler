import unittest
from scripts.diagnose_coverage_failures import classify


class FailureClassificationTests(unittest.TestCase):
    def test_mean_precedes_significance(self):
        self.assertEqual(classify(dict(n=40, mean=0, p_less=0)), 'mean_not_better')

    def test_family_threshold_not_unadjusted(self):
        self.assertEqual(classify(dict(n=40, mean=-1, p_less=.02)), 'negative_mean_not_family_significant')
        self.assertEqual(classify(dict(n=40, mean=-1, p_less=.0001)), 'pass')

    def test_missing_support(self):
        self.assertEqual(classify(dict(n=1, mean=-1, p_less=None)), 'insufficient_statistical_support')


if __name__ == '__main__':
    unittest.main()
