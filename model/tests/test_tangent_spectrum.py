import unittest

import numpy as np
import pandas as pd

from scripts.audit_tangent_spectrum import split_direction,cast_parts,descriptions,VARIANTS
from scripts import fit_tangent_bridge as parent


class TangentSpectrumTest(unittest.TestCase):
    def test_known_strong_weak_null(self):
        j=np.array([[2.,0.,0.],[0.,.1,0.]])
        parts,r=split_direction(j,np.array([3.,4.,5.]),.1)
        np.testing.assert_allclose(parts['strong'],[3.,0.,0.],atol=1e-14)
        np.testing.assert_allclose(parts['weak'],[0.,4.,0.],atol=1e-14)
        np.testing.assert_allclose(parts['null'],[0.,0.,5.],atol=1e-14)
        self.assertEqual((r['strong'],r['weak']),(1,1))

    def test_rotation_invariance(self):
        rng=np.random.default_rng(8);j=rng.normal(size=(8,12));w=rng.normal(size=12)
        q=np.linalg.qr(rng.normal(size=(12,12)))[0]
        p,_=split_direction(j,w,10.);other,_=split_direction(j@q,q.T@w,10.)
        for k in p:np.testing.assert_allclose(j@p[k],j@q@other[k],atol=1e-11)

    def test_collinear_and_zero(self):
        p,r=split_direction(np.array([[1.,2.],[2.,4.]]),np.array([1.,0.]),1.)
        self.assertEqual(r['retained'],1)
        np.testing.assert_allclose(sum(p.values()),[1.,0.],atol=1e-14)
        p,r=split_direction(np.zeros((2,3)),np.ones(3),0.)
        self.assertEqual(r['retained'],0)
        np.testing.assert_array_equal(p['null'],np.ones(3))

    def test_threshold_equality(self):
        p,r=split_direction(np.eye(2),np.ones(2),1.)
        self.assertEqual(r['strong'],2)
        np.testing.assert_array_equal(p['weak'],np.zeros(2))

    def test_ill_conditioned_projection_has_analytic_solution(self):
        rng=np.random.default_rng(17)
        left=np.linalg.qr(rng.normal(size=(32,32)))[0]
        right=np.linalg.qr(rng.normal(size=(48,48)))[0]
        singular=np.geomspace(1.,1e-4,32)
        j=(left*singular)@right[:,:32].T
        w=right@np.linspace(100.,500.,48)
        parts,report=split_direction(j,w,1e-4)
        strong=singular**2>=1e-4
        expected=right[:,:32][:,strong]@(right[:,:32][:,strong].T@w)
        np.testing.assert_allclose(parts['strong'],expected,rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(parts['null'],right[:,32:]@(right[:,32:].T@w),rtol=1e-8,atol=1e-8)
        self.assertLess(report['orthogonality_error'],1e-7)

    def test_invalid_inputs(self):
        for j,w,l in [(np.ones((2,3)),np.ones(2),1.),(np.eye(2),np.ones(2),-1.),
                (np.eye(2)*np.nan,np.ones(2),1.),(np.eye(2),np.ones(2)*np.inf,1.)]:
            with self.assertRaises(ValueError):split_direction(j,w,l)

    def test_scalar_weights_and_unsupported(self):
        f=pd.DataFrame([dict(patient=f'p{i//2}',component=f'c{i//4}',fold=i%2,context=0,
            variable='map',truth=float(i),local_linear=.2,
            **{n:.1 for n in (parent.PRIMARY,*parent.CONTROLS)},
            **{n:.3 for n in VARIANTS}) for i in range(6)])
        r,error=descriptions(f,['map','bun'])
        self.assertLess(error,1e-12)
        self.assertEqual(r['strong_actual']['bun']['status'],'unsupported')
        self.assertEqual(r['strong_actual']['map']['patients'],3)
        with self.assertRaises(ValueError):descriptions(f.assign(fold=0),['map'])
        f.loc[0,'strong_actual']=np.nan
        with self.assertRaises(ValueError):descriptions(f,['map'])

    def test_cast_residual_is_accounted_not_applied(self):
        parts=dict(strong=np.array([100000003.,1.]),weak=np.array([-100000000.,1.]),
                   null=np.zeros(2))
        direction=sum(parts.values())
        result=cast_parts(parts,direction)
        np.testing.assert_array_equal(sum(result.values()),direction)
        self.assertEqual(result['roundoff'][0],3.)
        for name in parts:
            np.testing.assert_array_equal(result[name],parts[name].astype(np.float32).astype(float))
        gradient=np.array([2.,.5])
        self.assertEqual(sum(float(gradient@v) for v in result.values()),float(gradient@direction))
        with self.assertRaises(AssertionError):cast_parts(parts,direction+1.)


if __name__=='__main__':unittest.main()
