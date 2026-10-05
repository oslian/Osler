import unittest
from scripts.diagnose_coverage_component_influence import influence


class InfluenceTests(unittest.TestCase):
    def test_equal_patient_distinction(self):
        r=influence([dict(patients=1,real_minus_process=2),dict(patients=100,real_minus_process=-1)],'process')
        self.assertGreater(r['mean'],0)
        self.assertLess(r['patient_weighted_mean'],0)
        self.assertEqual(r['leave_one_sign_changes'],1)
        self.assertEqual(r['supported_component_mean'],-1)

    def test_all_better(self):
        r=influence([dict(patients=40,real_minus_process=-2),dict(patients=50,real_minus_process=-1)],'process')
        self.assertLess(r['leave_one_mean_max'],0)
        self.assertEqual(r['fraction_better'],1)
        self.assertIsNone(r['small_component_mean'])

    def test_invalid_group(self):
        with self.assertRaises(ValueError):
            influence([dict(patients=0,real_minus_process=1),dict(patients=1,real_minus_process=0)],'process')


if __name__=='__main__':unittest.main()
