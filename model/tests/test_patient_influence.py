import unittest
from scripts.audit_potassium_patient_influence import influence


class PatientInfluenceTest(unittest.TestCase):
    def test_equal_patient_weight_and_single_case_reversal(self):
        r=influence(['a','a','a','b'],[0]*4,[0,0,0,2],[1,1,1,0])
        self.assertEqual(r['patient_mean'],.5)
        self.assertEqual(r['row_mean'],-.25)
        self.assertTrue(r['any_single_patient_reverses_sign'])
        self.assertEqual(r['leave_one_out_mean_range'],[-1.,2.])

    def test_uniform_gain_not_single_case(self):
        r=influence(['a','b','c'],[0]*3,[0]*3,[1]*3)
        self.assertFalse(r['any_single_patient_reverses_sign'])
        self.assertEqual(r['benefit_fraction'],1.)
        with self.assertRaises(ValueError):influence(['a'],[0],[0],[0])


if __name__=='__main__':unittest.main()
