import itertools
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.diagnose_bridge_geometry import geometry, scalar_geometry, summarize, classify


def frame():
    return pd.DataFrame(dict(patient=['a','a','b','c'], component=['x','x','x','z'],
        variable=['v']*4, fold=[0,1,0,0], context=[0]*4,
        truth=[2.,2.,-2.,4.], bound_start=[0.]*4,
        fixed_immediate=[1.,3.,1.,2.]))


class BridgeGeometryTests(unittest.TestCase):
    def test_exhaustive_piecewise_reference_and_torch_loss(self):
        for b, y, p in itertools.product([-2.,0.,.5,2.], repeat=3):
            actual = geometry([b], [p], [y]).iloc[0].to_dict()
            expected = scalar_geometry(b, p, y)
            for k in actual:
                self.assertAlmostEqual(actual[k], expected[k], places=12)
            loss = lambda value: float(F.smooth_l1_loss(
                torch.tensor(value, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)))
            self.assertAlmostEqual(actual['smooth_delta'], loss(p)-loss(b), places=12)
            self.assertAlmostEqual(actual['mae_delta'],
                actual['mae_right_derivative']+actual['crossing_curvature'], places=12)

    def test_right_derivatives_include_exact_zero_residual(self):
        for y, p in itertools.product([-2.,-.4,0.,.4,2.], [-3.,0.,3.]):
            row = geometry([0.], [p], [y]).iloc[0]
            epsilon = 1e-7
            self.assertAlmostEqual((abs(epsilon*p-y)-abs(y))/epsilon,
                row.mae_right_derivative, places=7)
            loss = lambda e: .5*e**2 if abs(e) < 1 else abs(e)-.5
            self.assertAlmostEqual((loss(epsilon*p-y)-loss(-y))/epsilon,
                row.smooth_right_derivative, places=5)

    def test_crossing_can_help_or_harm_and_zero_truth_costs(self):
        rows = geometry([0.,0.,0.], [1.5,3.,2.], [1.,1.,0.])
        np.testing.assert_allclose(rows.mae_delta, [-.5,1.,2.])
        np.testing.assert_allclose(rows.crossing_curvature, [1.,4.,0.])
        np.testing.assert_allclose(rows.away_harm, [0.,0.,2.])
        np.testing.assert_allclose(rows.crossing_harm, [0.,1.,0.])

    def test_patient_and_component_weights_not_row_weights(self):
        result, error = summarize(frame(), ['v','missing'], 'fixed_immediate')
        m = result['v']['metrics']
        self.assertAlmostEqual(m['occurrence']['mae_delta'], -.75)
        self.assertAlmostEqual(m['patient']['mae_delta'], -2/3)
        self.assertAlmostEqual(m['component']['mae_delta'], -1.)
        self.assertEqual(result['missing'], {'status':'unsupported'})
        self.assertLess(error, 1e-12)

    def test_scale_classification(self):
        f = frame().iloc[:1].copy()
        f['truth'] = 1.
        f['fixed_immediate'] = -1.
        r, _ = summarize(f, ['v'], 'fixed_immediate')
        self.assertEqual(classify(r, 'patient')['targets']['no_positive_uniform_scale_gain'], ['v'])
        f['fixed_immediate'] = 3.
        r, _ = summarize(f, ['v'], 'fixed_immediate')
        self.assertEqual(classify(r, 'patient')['targets']['crossing_driven_harm'], ['v'])

    def test_huber_gain_can_coexist_with_mae_harm(self):
        # Small wrong moves get less Huber weight; MAE still charges full magnitude.
        rows = geometry([0.,0.], [-.1,.08], [.01,2.])
        self.assertGreater(rows.mae_delta.mean(), 0.)
        self.assertLess(rows.smooth_delta.mean(), 0.)
        self.assertGreater(rows.mae_right_derivative.mean(), 0.)
        self.assertLess(rows.smooth_right_derivative.mean(), 0.)

    def test_convex_scale_claim_on_positive_mean_derivative(self):
        b, p, y = np.zeros(2), np.array([-2.,1.]), np.array([1.,2.])
        self.assertGreater(geometry(b,p,y).mae_right_derivative.mean(), 0.)
        for a in [.0001,.1,.5,1.,2.,10.]:
            self.assertGreater((abs(b+a*(p-b)-y)-abs(b-y)).mean(), 0.)

    def test_refuse_bad_numeric_inputs(self):
        for b,p,y in [([],[],[]), ([0.],[1.,2.],[1.]), ([0.],[np.nan],[1.]),
                      ([np.inf],[1.],[1.]), ([[0.]],[[1.]],[[1.]])]:
            with self.assertRaises(ValueError): geometry(b,p,y)

    def test_refuse_identity_inventory_and_channel_damage(self):
        f = frame()
        for bad in [pd.concat([f, f.iloc[:1]], ignore_index=True),
                    f.assign(patient=[None,'a','b','c']),
                    f.assign(component=['x','changed','x','z']),
                    f.assign(truth=[np.nan,2.,-2.,4.])]:
            with self.assertRaises(ValueError): summarize(bad,['v'],'fixed_immediate')
        with self.assertRaises(ValueError): summarize(f,['other'],'fixed_immediate')
        with self.assertRaises(ValueError): summarize(f,['v','v'],'fixed_immediate')
        with self.assertRaises(ValueError): summarize(f,['v'],'unknown')

    def test_unsupported_and_tiny_sign_are_not_success(self):
        f = frame().iloc[:1].copy()
        f['fixed_immediate'] = 0.
        r,_ = summarize(f, ['v','missing'], 'fixed_immediate')
        s = classify(r,'patient')
        self.assertEqual(s['supported'], 1)
        self.assertEqual(s['counts']['full_harm'], 0)
        r['v']['metrics']['patient']['mae_delta'] = 1e-15
        self.assertEqual(classify(r,'patient')['counts']['full_harm'], 0)


if __name__ == '__main__':
    unittest.main()
