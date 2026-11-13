import unittest

import torch

from osler_jepa.organ_schema import CANONICAL_ORGAN_VARIABLES
from scripts.audit_organ_assignment import audit, cases, pools, TOLERANCE
from scripts.audit_organ_assignment_support import support
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
from whole_body_joint_jepa import JointWholeBodyJEPAWorldModel


class OrganAssignmentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(104)
        self.variables = ('bun', 'creatinine', 'glucose')
        model = JointWholeBodyJEPAWorldModel(3, 14, 3, hidden=12,
            latent=8, token_dim=8, variable_module_ids=[0, 1, 2],
            horizon_values=(3.,), target_encoder_mode='fast')
        expand_history_projection(model, 6)
        model.belief_input_mode = 'observed_steps'
        self.trajectory = NativeOrganStateResidual(model, self.variables, ('creatinine',)).double().eval()
        self.model = self.trajectory.jepa

    def test_distinct_assignments_collide_before_and_after_native_evolution(self):
        records = audit(self.trajectory)
        self.assertEqual(len(records), 6)
        for row in records:
            for key in ('organ_proposal', 'organ_post_update', 'organ_after_6h', 'formula_parity'):
                self.assertLessEqual(row[key], TOLERANCE)
            for key in ('binding_proposal', 'binding_post_update', 'binding_after_6h'):
                self.assertGreater(row[key], TOLERANCE)
        self.assertTrue(any(row['body_proposal'] > TOLERANCE for row in records))

    def test_equal_values_are_not_required_to_separate(self):
        x, _, _ = cases(self.model, self.variables)
        x[:, :2] = 1
        y = x.clone(); y[:, [0, 1]] = x[:, [1, 0]]
        for a, b in zip(pools(self.model, self.variables, x), pools(self.model, self.variables, y)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_reorder_complete_records_not_numeric_assignments(self):
        x, _, _ = cases(self.model, self.variables)
        expected = pools(self.model, self.variables, x)
        variables = tuple(reversed(self.variables))
        n = len(variables); order = torch.arange(n-1, -1, -1)
        y = x.clone()
        for offset in (0, n, 2*n):
            y[:, offset:offset+n] = x[:, offset+order]
        with torch.no_grad():
            self.model.variable_embedding.copy_(self.model.variable_embedding[order].clone())
            self.model.variable_module_ids.copy_(self.model.variable_module_ids[order].clone())
        for a, b in zip(expected, pools(self.model, variables, y)):
            torch.testing.assert_close(a, b, rtol=0, atol=TOLERANCE)

    def test_missing_nan_is_not_an_observed_zero(self):
        x, _, _ = cases(self.model, self.variables)
        y = x.clone(); y[:, 2] = float('nan'); y[:, 8] = float('nan')
        for a, b in zip(pools(self.model, self.variables, x), pools(self.model, self.variables, y)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        y[:, 5] = 1; y[:, 2] = 0; y[:, 8] = 0
        _, present = pools(self.model, self.variables, y)
        pancreas = tuple(CANONICAL_ORGAN_VARIABLES).index('pancreas')
        self.assertGreater(float(present[:, pancreas].detach().abs().max()), 0)

    def test_unobserved_organ_update_stays_unchanged(self):
        x, _, _ = cases(self.model, self.variables)
        body, organs, presence = encode_native_organ_observations(self.model, self.variables, x)
        layer = self.trajectory.layer
        initial = layer.initialize(torch.randn_like(body), torch.randn_like(organs), torch.ones_like(presence))
        state = layer.update(initial, body, organs, presence, body.new_zeros(len(body)))
        torch.testing.assert_close(state.organs[presence == 0], initial.organs[presence == 0], rtol=0, atol=0)

    def test_support_requires_multiple_distinct_observations(self):
        x, _, _ = cases(self.model, self.variables)
        counts = support(x.detach(), self.variables)
        self.assertEqual(counts['distinct_events'], 6)
        self.assertEqual(counts['organs']['pancreas']['multi_events'], 0)
        x[:, :2] = 0
        self.assertEqual(support(x.detach(), self.variables)['distinct_events'], 0)
        x[:, 7] = 1
        self.assertEqual(support(x.detach(), self.variables)['distinct_events'], 6)
        x[:, 4] = 0
        self.assertEqual(support(x.detach(), self.variables)['multi_events'], 0)


if __name__ == '__main__':
    unittest.main()
