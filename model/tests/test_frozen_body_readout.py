import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scripts.refit_frozen_body_readout import objective,solve


class FrozenReadoutTests(unittest.TestCase):
    def test_gradient(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(15,3));y=rng.normal(size=15);w=np.ones(15)/15;t=np.array([.2,.1,-.2,.3])
        analytic=objective(t,x,y,w)[1]
        numeric=approx_derivative(lambda v:objective(v,x,y,w)[0],t).reshape(-1)
        np.testing.assert_allclose(analytic,numeric,atol=1e-7)

    def test_affine_recovery(self):
        x=np.arange(20,dtype=float)[:,None];y=2*x[:,0]+3
        weight,bias,report=solve(x,y,np.ones(20),np.zeros(1),0.)
        self.assertTrue(report['success']);np.testing.assert_allclose(x@weight+bias,y,atol=1e-5)
