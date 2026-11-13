import unittest
import pandas as pd
from scripts.audit_crossfit_offset_calibration import fit_offset


class OffsetCalibrationTests(unittest.TestCase):
    def test_replication_of_group_cannot_dominate_offset(self):
        a = pd.DataFrame(dict(group=['selected']*30, truth=[1.]*30, real=[0.]*30))
        b = pd.DataFrame(dict(group=['broad']*30, truth=[2.]*30, real=[0.]*30))
        x = fit_offset(pd.concat([a, b]))
        y = fit_offset(pd.concat([a, b, b, b]))
        self.assertEqual(x['offset'], y['offset'])

    def test_sparse_group_preserves_original(self):
        f = pd.DataFrame(dict(group=['selected']*29+['broad']*100,
                             truth=[3.]*129, real=[1.]*129))
        r = fit_offset(f)
        self.assertEqual(r['status'], 'insufficient_calibration')
        self.assertEqual(r['offset'], 0.)

    def test_offset_sign_corrects_constant_bias(self):
        f = pd.DataFrame(dict(group=['selected']*30+['broad']*30,
                             truth=[3.]*60, real=[1.]*60))
        self.assertEqual(fit_offset(f)['offset'], 2.)
