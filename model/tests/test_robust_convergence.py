import unittest
import numpy as np
from scripts.repair_robust_convergence import solve_more


class ConvergenceTests(unittest.TestCase):
    def test_same_objective_continuation_on_well_conditioned_data(self):
        rng=np.random.default_rng(19)
        x=rng.normal(size=(150,3));y=x[:,0]+rng.normal(size=150)*.2
        rows=solve_more(x,y,np.full(150,1/150),x)
        self.assertTrue(all(r['converged'] for r in rows))
        np.testing.assert_allclose(rows[0]['prediction'],rows[1]['prediction'],atol=1e-3)
        self.assertEqual(rows[1]['stage'],'continued')


if __name__=='__main__':unittest.main()
