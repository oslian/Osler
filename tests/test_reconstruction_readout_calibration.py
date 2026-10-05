import unittest
import numpy as np
from scripts.audit_reconstruction_readout_calibration import fit_affine


class CalibrationTests(unittest.TestCase):
    def test_recovers_positive_line(self):
        a,b=fit_affine(np.arange(10.), 3*np.arange(10.)+2)
        self.assertAlmostEqual(a,3)
        self.assertAlmostEqual(b,2)

    def test_no_negative_slope(self):
        a,_=fit_affine(np.arange(10.), -np.arange(10.))
        self.assertGreaterEqual(a,0)

    def test_constant(self):
        self.assertEqual(fit_affine(np.ones(3),np.array([1.,2.,100.])),(0.,2.))

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            fit_affine([1,np.nan],[2,3])
