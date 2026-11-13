import unittest
from scripts.audit_correction_weight_concentration import concentration


class WeightConcentrationTest(unittest.TestCase):
    def test_equal_groups(self):
        self.assertEqual(concentration([100, 100])['kish_weight_ess'], 200)

    def test_single_small_group(self):
        r = concentration([1, 10000])
        self.assertEqual(r['max_to_min_patient_weight'], 10000)
        self.assertEqual(r['small_component_weight_mass'], .5)
        self.assertLess(r['kish_weight_ess'], 4)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            concentration([0, 10])
