import unittest
from types import SimpleNamespace
import numpy as np
from scripts.audit_scaled_kernel_numerics import check_scaler


class ScalerNumericsTests(unittest.TestCase):
    def test_guarded_roundoff_and_invalid_negative_variance(self):
        s=SimpleNamespace(var_=np.array([-1e-24,0.,2.]),mean_=np.array([49.,0.,0.]),
                          scale_=np.array([1.,1.,np.sqrt(2.)]),n_samples_seen_=2000)
        self.assertEqual(check_scaler(s)['negative_variances'],1)
        s.var_[0]=-1.
        with self.assertRaises(ValueError): check_scaler(s)

    def test_invalid_scale_rejected(self):
        for scale in (np.nan,0.,-1.,np.inf):
            s=SimpleNamespace(var_=np.array([0.]),mean_=np.array([0.]),
                              scale_=np.array([scale]),n_samples_seen_=10)
            with self.assertRaises(ValueError): check_scaler(s)


if __name__=='__main__': unittest.main()
