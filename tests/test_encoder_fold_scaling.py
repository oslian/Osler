import unittest
import numpy as np
import torch
from scripts.encoder_fold_scaling import fit_scaler, transform_example, inverse_value, patient_centers


class FoldScalingTests(unittest.TestCase):
    def test_heldout_values_do_not_affect_scaler(self):
        x=np.array([[1.,2.],[3.,4.],[7.,9.]])
        a=fit_scaler(x,[0,1]);x[2]=1e10;b=fit_scaler(x,[0,1])
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_roundtrip_keeps_zero_and_missing_distinct(self):
        e=dict(target_index=0,truth_normalized=0.,tokens=torch.tensor([[0.,0.,1.,0.,.1,7.,9.]]))
        scaler=fit_scaler([[1.,2.],[3.,4.]],[0,1])
        q=transform_example(e,scaler)
        self.assertEqual(q['tokens'][0,0],-2.)
        self.assertEqual(q['tokens'][0,1],0.)
        torch.testing.assert_close(q['tokens'][:,2:],e['tokens'][:,2:])
        self.assertEqual(inverse_value(q['truth_normalized'],0,scaler),0.)

    def test_unseen_observation_is_not_silently_imputed(self):
        scaler=fit_scaler([[1.,np.nan],[2.,np.nan]],[0,1])
        e=dict(target_index=0,truth_normalized=1.,tokens=torch.tensor([[1.,2.,1.,1.,0.,0.]]))
        with self.assertRaisesRegex(ValueError,'Observed variable'):
            transform_example(e,scaler)

    def test_anchor_preferred_over_repeated_history(self):
        e=dict(target_index=0,truth_normalized=5.,tokens=torch.tensor([[10.,3.,1.,1.,0.,0.]]))
        np.testing.assert_array_equal(patient_centers({0:e},2),[5.,3.])
