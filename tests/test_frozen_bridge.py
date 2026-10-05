from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.train_frozen_bridge import CHANNELS, NEW, assess, describe, fit_bridge
from scripts.train_organ_assimilation import AssimilatedReconstruction
from scripts.train_content_retention import replay
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


class FrozenBridgeTest(unittest.TestCase):
    def model(self):
        torch.manual_seed(63)
        m = nn.Module(); m.trajectory = nn.Module()
        m.trajectory.layer = PersistentPhysiologyStateLayer(body_dim=8,organ_dim=4,
            organ_count=3,adjacency=torch.eye(3)).double()
        m.readout = nn.Linear(8,2).double()
        return m

    def test_gradients_update_only_projection_through_frozen_history(self):
        m = self.model(); saved = deepcopy(m.state_dict())
        e = dict(body=torch.randn(3,8,dtype=torch.double),organs=torch.randn(3,3,4,dtype=torch.double),
            presence=torch.ones(3,3,dtype=torch.double),times=torch.tensor([0.,1.,2.],dtype=torch.double),
            anchor=3.,target=0,truth=3.)
        for c in CHANNELS:
            zero = AssimilatedReconstruction(m,c)
            torch.testing.assert_close(zero(e),m.readout(replay(m.trajectory.layer,e))[0,0],atol=0,rtol=0)
            trained,log = fit_bridge(m,[e],[e['organs']],c)
            self.assertTrue(all(x['nonzero_gradients']==1 for x in log))
            self.assertGreater(float(trained.bridge.projection.weight.detach().abs().max()),0.)
            for k,v in saved.items():
                torch.testing.assert_close(trained.model.state_dict()[k],v,atol=0,rtol=0)
                torch.testing.assert_close(m.state_dict()[k],v,atol=0,rtol=0)
            self.assertEqual([k for k,p in trained.named_parameters() if p.requires_grad],['bridge.projection.weight'])
            self.assertTrue(all(p.grad is None for p in trained.model.parameters()))

    def frame(self):
        rows = []
        for p,fold,error in [('a',0,1.),('a',1,1.),('b',0,3.)]:
            r = dict(patient=p,component=p,fold=fold,context=0,variable='map',truth=0.)
            for c in (*NEW,'bound_start','immediate','delayed','body','unbridged','frozen','raw'): r[c]=error
            rows.append(r)
        return pd.DataFrame(rows)

    def test_patient_equal_and_unsupported(self):
        f=self.frame(); r=describe(f,['map','bun'])
        self.assertEqual(r['map']['normalized_mae']['patient']['fixed_immediate'],2.)
        self.assertEqual(r['bun'],dict(status='unsupported'))
        scored=assess(f,1,repetitions=20)
        self.assertTrue(all(not c['screen_pass'] for c in scored['comparisons'].values()))

    def test_invalid_inputs(self):
        f=self.frame()
        for broken in (pd.concat([f,f.iloc[:1]]),f.assign(fixed_immediate=np.nan),f.assign(patient=None)):
            with self.assertRaises(ValueError): describe(broken,['map'])
        with self.assertRaises(ValueError): assess(f.drop(columns=['raw']),1,repetitions=20)
        with self.assertRaises(ValueError): fit_bridge(self.model(),[],[],'immediate')


if __name__=='__main__': unittest.main()
