import unittest
import numpy as np
from scripts.audit_frozen_readout_spectrum import spectrum


class SpectrumTests(unittest.TestCase):
    def test_weak_axis_projection(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(2000,4))*[3,2,1,.001]
        mean,weak,report=spectrum(x,np.ones(len(x)),np.array([0.,0.,0.,1.]))
        self.assertEqual(report['weak_dimensions'],1)
        self.assertLess(report['weak_variance_fraction'],1e-5)
        np.testing.assert_allclose(weak,[0,0,0,1],atol=1e-3)
        self.assertEqual(mean.shape,(4,))
