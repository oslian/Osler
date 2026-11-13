import unittest
from scripts.audit_continued_selection_pairs import interval


class PairedIntervalTest(unittest.TestCase):
    def test_constant_and_zero(self):
        for v in (-2.,0.,3.):
            result=interval([v]*30,repetitions=200)
            self.assertEqual((result['mean'],result['lower'],result['upper']),(v,v,v))

    def test_reproducible_and_invalid(self):
        self.assertEqual(interval([-1,0,2],repetitions=200),interval([-1,0,2],repetitions=200))
        for v in ([],[float('nan')]):
            with self.assertRaises(ValueError):interval(v)
