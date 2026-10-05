import unittest
import numpy as np
from scripts.audit_bun_affine_transfer import fit_affine,arrays


class AffineTests(unittest.TestCase):
    def test_recovers_affine(self):
        x=np.arange(20,dtype=float)
        fit=fit_affine(x,2*x-3)
        self.assertAlmostEqual(fit['slope'],2.)
        self.assertAlmostEqual(fit['intercept'],-3.)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError): arrays([],'full')
