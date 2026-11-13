import unittest
import numpy as np
import torch
from tests.test_native_body_reconstruction import BodyReconstructionTest
from scripts.audit_reconstruction_stage_probe import choose, event_summaries, inputs


class StageProbeTests(unittest.TestCase):
    setUp = BodyReconstructionTest.setUp

    def test_sampling_order_independent(self):
        self.assertEqual(choose(['a','b','c'],2),choose(['c','b','a'],2))
        with self.assertRaises(ValueError): choose(['a','a'],1)

    def test_masked_target_not_in_event_features(self):
        e=dict(tokens=self.tokens,event_hours=self.times,anchor_hour=2.,target_index=0)
        a=event_summaries(self.model.trajectory.jepa,self.model.trajectory.variables,e)
        altered=self.tokens.clone(); altered[:,0]=1000.; altered[:,6]=100.
        b=event_summaries(self.model.trajectory.jepa,self.model.trajectory.variables,dict(e,tokens=altered))
        for left,right in zip(a,b): np.testing.assert_array_equal(left,right)

    def test_invalid_time_rejected_before_mask(self):
        with self.assertRaises(ValueError):
            event_summaries(self.model.trajectory.jepa,self.model.trajectory.variables,
                dict(tokens=self.tokens,event_hours=torch.tensor([0.,4.]),anchor_hour=2.,target_index=0))

    def test_equal_process_features(self):
        d=dict(raw=np.arange(530.).reshape(2,265),body=np.zeros((2,64)),
               event_mean=np.ones((2,64)),event_last=np.full((2,64),2.))
        r=inputs(d)
        for key in ('body','event_mean','event_last'):
            self.assertEqual(r[key].shape,(2,223))
            np.testing.assert_array_equal(r[key][:,64:],r['process'])
        d['body'][0,0]=np.nan
        with self.assertRaises(ValueError): inputs(d)
