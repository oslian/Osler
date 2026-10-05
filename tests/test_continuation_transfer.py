import unittest
from scripts.audit_continuation_transfer import category


class TransferSignsTest(unittest.TestCase):
    def test_signs_are_not_conflated(self):
        self.assertEqual(category(-1,1),'source_improves/evaluation_not_improved')
        self.assertEqual(category(1,-1),'source_not_improved/evaluation_improves')
        self.assertEqual(category(0,0),'source_not_improved/evaluation_not_improved')

    def test_nan_rejected(self):
        with self.assertRaises(ValueError): category(float('nan'),1)
