import unittest
import pandas as pd
from scripts.audit_crossfit_calibration_transfer import compare


class CalibrationTransferTests(unittest.TestCase):
    def test_opposite_bias_is_harm_not_deployable_oracle(self):
        f = pd.DataFrame(dict(group=['selected']*30+['broad']*30,
                              truth=[0.]*60, real=[1.]*60))
        r = compare(f, dict(offset=1., status='calibrated'))
        self.assertTrue(r['opposite_direction'])
        self.assertEqual(r['equal_group_delta'], 1.)
        self.assertEqual(r['equal_group_mae']['retrospective_oracle'], 0.)

    def test_insufficient_oracle_not_zero_bias_evidence(self):
        f = pd.DataFrame(dict(group=['selected']*2+['broad']*30,
                              truth=[0.]*32, real=[1.]*32))
        r = compare(f, dict(offset=1., status='calibrated'))
        self.assertFalse(r['supported_both'])
        self.assertIsNone(r['opposite_direction'])
        self.assertIsNone(r['equal_group_mae']['retrospective_oracle'])
