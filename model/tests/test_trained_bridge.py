from copy import deepcopy
import unittest

import pandas as pd
import torch
from torch import nn

from scripts.audit_trained_bridge import disable_bridge, summarize
from scripts.organ_assimilation import CHANNELS
from scripts.train_content_retention import replay
from scripts.train_organ_assimilation import AssimilatedReconstruction
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


class TrainedBridgeTest(unittest.TestCase):
    def test_intervention_changes_only_projection_and_matches_native(self):
        torch.manual_seed(55)
        original = nn.Module()
        original.trajectory = nn.Module()
        original.trajectory.layer = PersistentPhysiologyStateLayer(body_dim=8,
            organ_dim=4, organ_count=3, adjacency=torch.eye(3)).double()
        original.readout = nn.Linear(8, 2).double()
        e = dict(body=torch.randn(3, 8, dtype=torch.double),
            organs=torch.randn(3, 3, 4, dtype=torch.double),
            presence=torch.tensor([[1., 1., 1.], [0., 0., 0.], [1., 0., 1.]], dtype=torch.double),
            times=torch.tensor([0., 1., 3.], dtype=torch.double), anchor=4., target=0)
        for channel in CHANNELS:
            candidate = AssimilatedReconstruction(original, channel)
            with torch.no_grad(): candidate.bridge.projection.weight.normal_(0., .2)
            before = deepcopy(candidate.state_dict())
            off = disable_bridge(candidate)
            with torch.no_grad():
                expected = off.model.readout(replay(off.model.trajectory.layer, e))[0, 0]
                torch.testing.assert_close(off(e), expected, rtol=0, atol=0)
                self.assertNotEqual(float(candidate(e)), float(off(e)))
            for k, v in candidate.state_dict().items():
                torch.testing.assert_close(v, before[k], atol=0, rtol=0)
            self.assertTrue(all(not p.requires_grad for p in off.parameters()))

    def frame(self):
        rows = []
        for p, fold, component, on, off in [('a', 0, 'x', 1., 3.), ('a', 1, 'x', 1., 3.),
                                           ('b', 0, 'y', 3., 1.)]:
            r = dict(patient=p, fold=fold, component=component, context=0,
                variable='map', truth=0., unbridged=1., frozen=2.)
            for c in CHANNELS: r.update({c: on, c+'_off': off})
            rows.append(r)
        return pd.DataFrame(rows)

    def test_equal_patient_not_occurrence_and_cancellation(self):
        result = summarize(self.frame(), ['map', 'bun'])
        self.assertEqual(result['bun']['status'], 'unsupported')
        self.assertEqual(result['map']['normalized_mae']['patient']['immediate'], 2.)
        c = result['map']['contrasts']['immediate']['patient']
        self.assertEqual(c, dict(bridge=0., remaining=1., total=1.))
        one = summarize(self.frame().iloc[:2], ['map'])['map']['contrasts']['immediate']['patient']
        self.assertEqual(one, dict(bridge=-2., remaining=2., total=0.))

    def test_invalid_predictions_and_identities_rejected(self):
        f = self.frame()
        for broken in (pd.concat([f, f.iloc[:1]]), f.assign(immediate_off=float('nan')),
                       f.assign(raw=float('nan')), f.assign(patient=None), f.assign(variable='outside')):
            with self.assertRaises(ValueError): summarize(broken, ['map'])
        f.loc[1, 'component'] = 'changed'
        with self.assertRaises(ValueError): summarize(f, ['map'])


if __name__ == '__main__': unittest.main()
