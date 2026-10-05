import unittest
import numpy as np
from scripts.audit_dynamics_finite_step import make_directions


class FiniteStepTests(unittest.TestCase):
    def test_controls_have_matched_norm_and_descent_sign(self):
        tasks=['selected/bun','broad/bun','selected/other']
        a=np.array([[1.,-1.],[1.,-1.],[0.,1.]])
        d=make_directions(tasks,[a,a])
        for i in (0,1):
            p,u=d[f'probe{i}_protected'],d[f'probe{i}_matched_bun']
            self.assertAlmostEqual(np.linalg.norm(p),np.linalg.norm(u))
            self.assertGreater(float(a[0]@p),0)
            self.assertGreaterEqual(float(a[2]@p),-1e-6)
