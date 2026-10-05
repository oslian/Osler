import unittest
import numpy as np
from scripts.train_auxiliary_encoder import shuffled_deltas


class AuxiliaryPlaceboTest(unittest.TestCase):
    def test_distribution_and_fit_boundary(self):
        subjects=np.array(['a','a','b','b','c','c','holdout'])
        delta=np.arange(7.).reshape(-1,1)
        fit=np.array([True]*6+[False])
        result=shuffled_deltas(delta,np.ones_like(delta,dtype=bool),subjects,fit)
        np.testing.assert_equal(np.sort(result[fit,0]),np.sort(delta[fit,0]))
        np.testing.assert_equal(result[~fit],delta[~fit])
        for row in np.flatnonzero(fit):
            self.assertNotEqual(subjects[row],subjects[int(result[row,0])])

    def test_dominant_patient_rejected(self):
        with self.assertRaises(ValueError):
            shuffled_deltas(np.ones((3,1)),np.ones((3,1),bool),np.array(['a','a','b']),np.ones(3,bool))

    def test_unselected_sparse_column_is_ignored(self):
        delta=np.arange(12.).reshape(6,2)
        valid=np.ones((6,2),bool); valid[:,1]=False
        result=shuffled_deltas(delta,valid,np.array(['a','a','b','b','c','c']),np.ones(6,bool))
        np.testing.assert_equal(result[:,1],delta[:,1])

    def test_auxiliary_loss_respects_empty_labels(self):
        import torch
        from scripts.train_auxiliary_encoder import auxiliary_loss
        value=torch.ones(2,3,requires_grad=True)
        loss=auxiliary_loss(None,{'value':value},None,None,torch.zeros(2,3,dtype=torch.bool),[0,2])
        self.assertEqual(float(loss.detach()),0)
        loss.backward()
        torch.testing.assert_close(value.grad,torch.zeros_like(value))
