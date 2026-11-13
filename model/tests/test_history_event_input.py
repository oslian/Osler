import unittest
from types import SimpleNamespace
import numpy as np
import torch
from osler_jepa.patient_state import NeuralBeliefFilter
from whole_body_jepa.history_event_input import prepare_events,append_events,expand_history_projection


class HistoryEventInputTest(unittest.TestCase):
    def test_frozen_initial_prediction_and_clock(self):
        torch.manual_seed(7)
        model=SimpleNamespace(belief_filter=NeuralBeliefFilter(5,8),input_dim=5)
        x=torch.randn(2,3,5);mask=torch.ones(2,3,dtype=torch.bool)
        old=model.belief_filter(x,mask)[0].detach()
        extra=np.ones((6,6),np.float32)
        joined=append_events({'input':x.reshape(6,5).numpy()},extra)['input']
        np.testing.assert_array_equal(joined[:,-1],x.reshape(6,5).numpy()[:,-1])
        expand_history_projection(model,6)
        actual=model.belief_filter(torch.from_numpy(joined).reshape(2,3,11),mask)[0]
        torch.testing.assert_close(old,actual,atol=1e-6,rtol=1e-6)
        actual.sum().backward()
        self.assertGreater(float(model.belief_filter.observation_projection[0].weight.grad[:,4:10].abs().sum()),0)

    def test_source_scaling_and_zero(self):
        raw=np.array([[0.,np.nan],[2.,4.],[1000.,9000.]])
        x,c,s=prepare_events(raw,np.array([0,1]))
        np.testing.assert_array_equal(c,[1.,4.])
        self.assertEqual(x[0,2],1.)
        self.assertEqual(x[0,3],0.)
        self.assertEqual(x[0,1],0.)


if __name__=='__main__':unittest.main()
