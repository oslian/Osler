import unittest
import numpy as np
from scripts.refit_anchored_body_readout import fit,source_weights,source_score


class AnchoredTests(unittest.TestCase):
    def test_large_penalty_stays_near_parent(self):
        x=np.arange(20,dtype=float)[:,None];y=2*x[:,0]+3
        w,b=fit(x,y,np.ones(20),np.array([1.]),0.,1e6)
        np.testing.assert_allclose(x@w+b,x[:,0],atol=1e-4)

    def test_equal_group_mass(self):
        groups=np.array(['a','b','b']);sites=np.array(['s','s','t'])
        w=source_weights(groups,sites)
        self.assertAlmostEqual(w[0],.5);self.assertAlmostEqual(w[1:].sum(),.5)
        self.assertAlmostEqual(source_score(np.array([2.,0.,4.]),groups,sites),2.)
