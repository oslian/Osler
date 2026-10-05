import unittest
import numpy as np
from scripts.diagnose_kernel_objective import spectrum, ray_audit, dispersion


class KernelObjectiveTests(unittest.TestCase):
    def test_spectrum_rotation_and_constant(self):
        rng=np.random.default_rng(3); x=rng.normal(size=(100,5)); w=np.linspace(.5,2,100)
        q,_=np.linalg.qr(rng.normal(size=(5,5)))
        a,b=spectrum(x,w),spectrum(x@q,w)
        self.assertAlmostEqual(a['effective_rank'],b['effective_rank'],places=10)
        self.assertEqual(a['variance95_rank'],b['variance95_rank'])
        self.assertEqual(spectrum(np.zeros((10,5)),np.ones(10))['effective_rank'],0)

    def test_known_regularized_optimum_is_not_lad_optimum(self):
        # J(s)=.5*s^2+|s-3|; optimum1, but MAE alone prefers3.
        r=ray_audit(np.ones(10),np.ones(10)*3,np.ones(10),1.,.1)
        self.assertEqual(r['lad_gain'],3.)
        self.assertEqual(r['lad_weighted_mae'],0.)
        self.assertGreater(r['lad_objective'],r['current_objective'])
        self.assertAlmostEqual(r['ray_objective_gain'],1.,places=6)
        self.assertLess(r['ray_relative_objective_reduction'],1e-12)

    def test_ray_detects_nonoptimal_solution_and_zero_prediction(self):
        r=ray_audit(np.ones(10)*.5,np.ones(10)*3,np.ones(10),.25,.1)
        self.assertAlmostEqual(r['ray_objective_gain'],2.,places=6)
        self.assertGreater(r['ray_relative_objective_reduction'],.01)
        z=ray_audit(np.zeros(5),np.ones(5),np.ones(5),0.,1.)
        self.assertEqual(z['lad_gain'],0.)
        self.assertEqual(z['current_weighted_mae'],z['lad_weighted_mae'])

    def test_weighted_signed_ray_and_invalid_inputs(self):
        r=ray_audit(np.array([-1.,2.,0.]),np.array([-2.,4.,8.]),np.array([1.,2.,1.]),1.,1.)
        self.assertEqual(r['lad_gain'],2.)
        self.assertEqual(r['lad_weighted_mae'],2.)
        with self.assertRaises(ValueError): ray_audit([1],[2],[0],1,1)
        with self.assertRaises(ValueError): spectrum(np.ones((3,2)),np.ones(2))

    def test_dispersion_constant_truth_is_undefined_not_zero(self):
        d=dispersion(np.ones(10),np.ones(10),np.ones(10))
        self.assertIsNone(d['weighted_std_ratio'])
        self.assertIsNone(d['weighted_correlation'])


if __name__=='__main__': unittest.main()
