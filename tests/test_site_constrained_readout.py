import unittest
import numpy as np
from scripts.refit_site_constrained_readout import constraints, losses, fit_constrained


class ConstraintTests(unittest.TestCase):
    def test_constraint_derivative(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(20,3));y=rng.normal(size=20);theta=np.array([.1,.2,.3]);masks=[np.arange(20)<10,np.arange(20)>=10];reference=np.array([2.,2.])
        _,jac=constraints(theta,x,y,masks,reference)
        for i in range(3):
            delta=np.zeros(3);delta[i]=1e-6
            numeric=(constraints(theta+delta,x,y,masks,reference)[0]-constraints(theta-delta,x,y,masks,reference)[0])/2e-6
            np.testing.assert_allclose(jac[:,i],numeric,atol=1e-8)

    def test_no_sacrifice_of_perfect_site(self):
        x=np.zeros((40,1));y=np.r_[np.zeros(20),np.ones(20)]
        groups=np.repeat('a',40);sites=np.repeat(['s1','s2'],20)
        w,b,info=fit_constrained(x,y,groups,sites,np.zeros(1),0.)
        self.assertTrue(info['accepted'])
        self.assertLessEqual(losses(x[:20]@w+b-y[:20]).mean(),1e-8)
        self.assertLess(abs(b),1e-3)
