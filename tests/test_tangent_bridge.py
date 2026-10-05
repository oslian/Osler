from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
from torch import nn

from scripts.fit_tangent_bridge import (solve_tangent,huber,prediction_gradient,
    point_at,check_derivative,fit_step,describe,scalar_check,CONTROLS)
from scripts.train_organ_assimilation import AssimilatedReconstruction
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


class TangentBridgeTest(unittest.TestCase):
    def test_direct_primal_solution_and_nullspace(self):
        rng=np.random.default_rng(19)
        j=rng.normal(size=(14,7)); j=np.column_stack((j,j[:,0],np.zeros(14)))
        r=rng.normal(size=14)*2
        w,info=solve_tangent(j,r); lam=info['regularization']
        def objective(x):
            err=j@x-r
            return (huber(err).sum()+lam*x@x/2)/len(r),(j.T@np.clip(err,-1,1)+lam*x)/len(r)
        ref=minimize(objective,np.zeros(j.shape[1]),jac=True,method='L-BFGS-B',
            options=dict(gtol=1e-10,ftol=1e-14,maxiter=2000,maxls=50))
        self.assertTrue(ref.success)
        self.assertAlmostEqual(objective(w)[0],ref.fun,places=8)
        np.testing.assert_allclose(j@w,j@ref.x,atol=1e-5,rtol=1e-5)
        self.assertEqual(w[-1],0)
        self.assertAlmostEqual(w[0],w[-2],places=9)

    def test_full_row_space_can_fit_but_identical_inputs_conflict(self):
        w,info=solve_tangent(np.eye(4),np.array([.1,-.3,.5,2.]))
        self.assertLess(info['final_objective'],.01*info['start_objective'])
        w,info=solve_tangent(np.ones((2,3)),np.array([1.,-1.]))
        np.testing.assert_allclose(w,0,atol=1e-12)
        self.assertEqual(info['retained_dimension'],1)
        self.assertEqual(info['final_objective'],info['start_objective'])

    def test_zero_and_invalid_inputs(self):
        w,info=solve_tangent(np.zeros((3,2)),np.ones(3))
        np.testing.assert_array_equal(w,[0,0]); self.assertEqual(info['retained_dimension'],0)
        for j,r in (([],[]),(np.ones((3,2)),[1,2]),(np.array([[np.nan]]),[1]),(np.ones((1,2)),[np.inf])):
            with self.assertRaises(ValueError):solve_tangent(j,r)

    def test_scaled_quadratic_matches_exact_ridge(self):
        rng=np.random.default_rng(82)
        u,_=np.linalg.qr(rng.normal(size=(24,24)))
        v,_=np.linalg.qr(rng.normal(size=(30,24)))
        j=(u*np.geomspace(.001,100.,24))@v.T
        residual=rng.normal(size=24)*.01
        w,info=solve_tangent(j,residual)
        expected=j.T@np.linalg.solve(j@j.T+info['regularization']*np.eye(24),residual)
        np.testing.assert_allclose(j@w,j@expected,atol=1e-6,rtol=1e-5)
        self.assertLess(info['gradient_inf'],1e-5)

    def candidate(self):
        torch.manual_seed(71)
        m=nn.Module();m.trajectory=nn.Module()
        m.trajectory.layer=PersistentPhysiologyStateLayer(body_dim=8,organ_dim=4,
            organ_count=3,adjacency=torch.eye(3))
        m.readout=nn.Linear(8,2)
        c=AssimilatedReconstruction(m,'immediate').eval().requires_grad_(False)
        e=dict(body=torch.randn(3,8),organs=torch.randn(3,3,4),presence=torch.ones(3,3),
            times=torch.tensor([0.,1.123456789,2.1],dtype=torch.float64),anchor=3.,target=0,truth=1.)
        return c,e

    def test_full_jacobian_real_recurrence_and_inference_generator(self):
        c,e=self.candidate();saved=deepcopy(c.state_dict())
        value,g=prediction_gradient(c,e)
        d=np.random.default_rng(4).normal(size=len(g))
        self.assertLess(check_derivative(deepcopy(c).double(),e,g,d),2e-5)
        self.assertEqual(value,point_at(c,e,np.zeros_like(g)))
        with torch.inference_mode():
            inferred={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in e.items()}
            actual,gg=prediction_gradient(c,inferred)
            self.assertEqual(actual,value);np.testing.assert_array_equal(gg,g)
        for k,v in c.state_dict().items():torch.testing.assert_close(v,saved[k],atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in c.parameters()))

    def test_fit_only_backtracking_zero_and_descent(self):
        c,e=self.candidate();b,g=prediction_gradient(c,e)
        truth=np.array([b+.1]);d=g/np.dot(g,g)*.1
        alpha,points,log=fit_step(c,[e],d,np.array([g@d]),np.array([b]),truth,1e-8)
        self.assertGreater(alpha,0); self.assertLess(abs(points[0]-truth[0]),.1)
        alpha,points,log=fit_step(c,[e],-d,np.array([-g@d]),np.array([b]),truth,1e-8)
        self.assertEqual(alpha,0); self.assertEqual(points[0],b)
        with self.assertRaises(ValueError):fit_step(c,[],d,np.array([0]),np.array([b]),truth,1.)

    def test_scalar_metrics_patient_not_fold_weighting(self):
        records=[]
        for p,component,fold,point in [('a','x',0,0),('a','x',1,0),('b','y',0,3)]:
            r=dict(patient=p,component=component,fold=fold,context=0,variable='map',truth=1.,
                tangent_step=float(point),local_linear=.5)
            r.update({c:0. for c in CONTROLS});records.append(r)
        f=pd.DataFrame(records)
        result=describe(f,['map','bun'])
        self.assertEqual(result['map']['mae']['patient']['tangent_step'],1.5)
        self.assertEqual(result['bun']['status'],'unsupported')
        self.assertLess(scalar_check(f,['map','bun']),1e-12)
        f.loc[0,'truth']=np.nan
        with self.assertRaises(ValueError):describe(f,['map'])


if __name__=='__main__':unittest.main()
