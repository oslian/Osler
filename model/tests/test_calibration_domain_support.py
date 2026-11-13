import unittest
from scripts.preflight_calibration_domain_support import partitions, supported_domains


class CalibrationDomainSupportTests(unittest.TestCase):
    def test_three_and_three_disjoint(self):
        choices=partitions(range(6))
        self.assertEqual(len(choices),20)
        for a,b in choices:
            self.assertFalse(a&b)
            self.assertEqual(a|b,set(range(6)))
            self.assertEqual(len(a),3)
            self.assertEqual(len(b),3)

    def test_five_domains_cannot_supply_two_three_domain_roles(self):
        self.assertEqual(partitions(range(5)),[])

    def test_larger_calibration_sets_are_not_omitted(self):
        choices=partitions(range(7))
        self.assertEqual(len(choices),70)
        self.assertEqual({len(a) for a,b in choices},{3,4})

    def test_patient_count_not_domain_diversity(self):
        self.assertEqual(supported_domains({'a':10000,'b':2,'c':1}, {'a','b','c'}),1)
