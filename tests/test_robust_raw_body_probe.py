import unittest
import numpy as np
from sklearn.linear_model import Ridge
from scripts.audit_robust_raw_body_probe import fit_smooth


class RobustProbeTests(unittest.TestCase):
    def test_quadratic_region_matches_ridge(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(100,4));y=.02*x[:,0]+.01*rng.normal(size=100);w=np.linspace(.5,1.5,100)
        a,b,_=fit_smooth(x,y,w)
        reference=Ridge(alpha=100.,solver='svd').fit(x,y,sample_weight=w)
        np.testing.assert_allclose(x@a+b,reference.predict(x),atol=1e-6)

    def test_single_outlier_does_not_set_constant_mean(self):
        x=np.zeros((101,2));y=np.r_[np.zeros(100),1000.]
        _,b,_=fit_smooth(x,y,np.ones(101))
        self.assertLess(abs(b),.02)
