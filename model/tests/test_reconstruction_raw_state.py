import unittest
import numpy as np
import torch
from scripts.audit_reconstruction_raw_state import summarize


class RawStateTests(unittest.TestCase):
    def test_mask_zero_and_lag(self):
        variables=['glucose','creatinine','bun']
        x=torch.tensor([[100.,0.,0.,1.,1.,0.,0.,0.,7.],
                        [200.,4.,0.,1.,1.,0.,0.,0.,7.]])
        raw,process=summarize(x,torch.tensor([1.,3.]),4.,variables,'glucose')
        np.testing.assert_array_equal(raw[:3],[0.,4.,0.])
        np.testing.assert_array_equal(raw[3:6],[0.,2.,0.])
        np.testing.assert_array_equal(raw[6:9],[0.,1.,0.])
        np.testing.assert_array_equal(raw[9:12],[49.,1.,49.])
        np.testing.assert_array_equal(process[:6],np.zeros(6))
        np.testing.assert_array_equal(process[6:],raw[6:])
        self.assertEqual(float(x[0,0]),100.)

    def test_target_numbers_cannot_affect_summary(self):
        x=torch.tensor([[1.,2.,1.,1.,0.,0.]])
        a,_=summarize(x,torch.tensor([0.]),1.,['glucose','creatinine'],'glucose')
        x[0,0]=10000
        b,_=summarize(x,torch.tensor([0.]),1.,['glucose','creatinine'],'glucose')
        np.testing.assert_array_equal(a,b)
