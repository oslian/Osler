import unittest
import numpy as np
from scripts.audit_current_raw_body_probe import project_fit


class ProjectionTests(unittest.TestCase):
    def test_query_does_not_change_fitting_projection(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(100,80));q=rng.normal(size=(5,80));w=np.ones(100)
        a,b=project_fit(x,q,w,True)
        aa,bb=project_fit(x,np.r_[q,np.full((1,80),1000.)],w,True)
        np.testing.assert_allclose(a,aa);np.testing.assert_allclose(b,bb[:5])
        self.assertEqual(a.shape,(100,64))

    def test_wrong_body_width_rejected(self):
        with self.assertRaises(ValueError):project_fit(np.ones((100,63)),np.ones((2,63)),np.ones(100),False)
