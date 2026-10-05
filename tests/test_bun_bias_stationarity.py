import unittest
from scripts.audit_bun_bias_stationarity import optimum


class StationarityTests(unittest.TestCase):
    def test_offset(self):
        shift,before,after=optimum([2.,2.],[1.,3.])
        self.assertAlmostEqual(shift,-2.)
        self.assertEqual(before,1.)
        self.assertAlmostEqual(after,0.)

    def test_invalid(self):
        with self.assertRaises(ValueError):optimum([1.],[0.])
