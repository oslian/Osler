import unittest
import numpy as np
import torch
from scripts.encoder_fold_support import supported_example


class FoldSupportTests(unittest.TestCase):
    def test_unseen_value_and_event_do_not_enter_encoder(self):
        scaler=dict(counts=np.array([2,0]),location=np.zeros(2),scale=np.ones(2))
        e=dict(target_index=0,truth_normalized=1.,event_hours=torch.tensor([1.,2.]),
            tokens=torch.tensor([[0.,7.,0.,1.,7.,0.],[0.,8.,1.,1.,0.,0.]]))
        a=supported_example(e,scaler)
        changed=dict(e,tokens=e['tokens'].clone())
        changed['tokens'][:,1]=1000
        b=supported_example(changed,scaler)
        torch.testing.assert_close(a['tokens'],b['tokens'])
        torch.testing.assert_close(a['event_hours'],torch.tensor([2.]))
        self.assertEqual(a['tokens'][0,0],0.)
        self.assertEqual(a['tokens'][0,2],1.)
        self.assertEqual(a['tokens'][0,3],0.)

    def test_target_without_fit_support_stays_unavailable(self):
        scaler=dict(counts=np.array([2,0]),location=np.zeros(2),scale=np.ones(2))
        e=dict(target_index=1,truth_normalized=2.,event_hours=torch.tensor([1.]),
            tokens=torch.tensor([[1.,0.,1.,0.,0.,7.]]))
        with self.assertRaisesRegex(ValueError,'Target has no source'):
            supported_example(e,scaler)
