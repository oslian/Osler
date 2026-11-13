import unittest

import torch
from torch import nn

from scripts.audit_organ_reachability import check_path, timing_support
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


class OrganReachabilityTest(unittest.TestCase):
    def test_final_organ_update_is_delayed_but_body_bypass_is_live(self):
        torch.manual_seed(9)
        model = nn.Module()
        model.trajectory = nn.Module()
        model.trajectory.layer = PersistentPhysiologyStateLayer(
            body_dim=8, organ_dim=4, organ_count=2, adjacency=torch.eye(2)).double()
        model.readout = nn.Linear(8, 3).double()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = check_path(model)
        for key in ('immediate_body_difference', 'immediate_prediction_difference',
                    'immediate_organ_gradient'):
            self.assertEqual(result[key], 0.)
        self.assertFalse(result['immediate_organ_path_exists'])
        for key in ('immediate_organ_difference', 'immediate_body_gradient',
                    'one_hour_body_difference', 'one_hour_prediction_difference',
                    'one_hour_organ_gradient'):
            self.assertGreater(result[key], 0.)
        self.assertTrue(result['one_hour_organ_path_exists'])
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_support_keeps_zero_gap_and_unmeasured_targets(self):
        e = dict(times=torch.tensor([0., 1.]), anchor=1., target=0,
                 organs=torch.zeros(2, 2, 4), presence=torch.tensor([[1., 0.], [1., 0.]]))
        b = torch.ones_like(e['organs'])
        result = timing_support([e, dict(e, anchor=2.), dict(e, times=torch.tensor([1.]),
            organs=e['organs'][-1:], presence=e['presence'][-1:])], [b, b, b[-1:]], ['a', 'b'])
        self.assertEqual(result['total'], dict(contexts=3, zero_gap=2,
            single_event_zero_gap=1, zero_gap_with_changed_last_organ=2))
        self.assertEqual(result['targets']['b']['contexts'], 0)
        absent_only = e['organs'].clone(); absent_only[:, 1] = 2
        self.assertEqual(timing_support([e], [absent_only], ['a'])['total']['zero_gap_with_changed_last_organ'], 0)
        for invalid in (dict(e, anchor=.5), dict(e, anchor=float('nan')),
                        dict(e, times=torch.tensor([1., 1.]))):
            with self.assertRaises(ValueError): timing_support([invalid], [b], ['a'])
        with self.assertRaises(ValueError): timing_support([e], [], ['a'])


if __name__ == '__main__':
    unittest.main()
