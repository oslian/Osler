import unittest
from scripts.audit_expanded_reconstruction_patient_alignment import alignment


class AlignmentTest(unittest.TestCase):
    def test_constant_has_no_alignment(self):
        result=alignment([1,2,8],[3,3,3])
        self.assertAlmostEqual(result['alignment_benefit'],0)

    def test_perfect_prediction_loses_with_other_patient(self):
        result=alignment([1,3,5],[1,3,5])
        self.assertEqual(result['original_mae'],0)
        self.assertAlmostEqual(result['expected_wrong_patient_mae'],8/3)

    def test_invalid(self):
        with self.assertRaises(ValueError):alignment([1],[1])
        with self.assertRaises(ValueError):alignment([1,2],[1,float('nan')])
