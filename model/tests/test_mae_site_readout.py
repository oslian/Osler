import unittest
import numpy as np
from scripts.refit_mae_site_readout import fit_mae


class MAESiteTests(unittest.TestCase):
    def test_exact_affine_recovery_without_penalty(self):
        x=np.arange(20,dtype=float).reshape(-1,1);y=2*x[:,0]+3
        groups=np.repeat('a',20);sites=np.repeat(['x','y'],10)
        w,b,info=fit_mae(x,y,groups,sites,np.zeros(1),0.,False,alpha=0.)
        np.testing.assert_allclose(x@w+b,y,atol=1e-7)

    def test_perfect_site_cannot_be_sacrificed(self):
        x=np.zeros((40,1));y=np.r_[np.zeros(10),np.ones(30)]
        groups=np.repeat('a',40);sites=np.r_[np.repeat('x',10),np.repeat('y',30)]
        w,b,info=fit_mae(x,y,groups,sites,np.zeros(1),0.,True)
        self.assertLess(abs(b),1e-7)
        self.assertLessEqual(info['max_site_mae_change'],1e-7)
