import unittest
import numpy as np
import torch
from scripts.audit_crossfit_variable_relations import latest_observations, relation


class VariableRelationTests(unittest.TestCase):
    def test_latest_available_not_missing_or_zero_imputation(self):
        e = dict(tokens=torch.tensor([[2., 0., 1., 0., 0., 7.],
                                      [0., 0., 1., 0., .5, 7.]]),
                 event_hours=torch.tensor([1., 2.]), anchor_hour=3.)
        scaler = dict(counts=np.array([2, 0]), scale=np.array([3., 1.]),
                      location=np.array([4., 0.]))
        v, a = latest_observations(e, scaler)
        self.assertEqual(v[0], 4.)
        self.assertEqual(a[0], 13.)
        self.assertTrue(np.isnan(v[1]))

    def test_sparse_or_constant_is_not_association(self):
        x = np.arange(29, dtype=float)
        self.assertIsNone(relation(x, x*0, x)['rho'])
        self.assertIsNone(relation(np.ones(30), np.zeros(30), np.arange(30))['rho'])

    def test_monotone_relation_and_missing_denominator(self):
        x = np.r_[np.arange(30.), np.nan]
        r = relation(x, np.zeros(31), np.arange(31.))
        self.assertAlmostEqual(r['rho'], 1.)
        self.assertEqual(r['observed'], 30)
        self.assertAlmostEqual(r['observed_fraction'], 30/31)
