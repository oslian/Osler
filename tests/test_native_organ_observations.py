import unittest
import torch
from whole_body_joint_jepa import JointWholeBodyJEPAWorldModel
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from osler_jepa.organ_schema import CANONICAL_ORGAN_VARIABLES


class NativeOrganObservationTest(unittest.TestCase):
    def setUp(self):
        self.variables = ('map', 'creatinine', 'glucose')
        self.model = JointWholeBodyJEPAWorldModel(3, 14, 3, hidden=12,
            latent=8, token_dim=8, variable_module_ids=[0, 1, 2],
            horizon_values=(3.,), target_encoder_mode='fast')
        expand_history_projection(self.model, 6)
        self.model.belief_input_mode = 'observed_steps'; self.model.eval()
        self.tokens = torch.zeros(2, 20)
        self.tokens[:, 3:6] = 1

    def test_local_organ_proposal_does_not_mix_other_organs(self):
        base = encode_native_organ_observations(self.model, self.variables, self.tokens)
        changed = self.tokens.clone(); changed[:, 2] = 10
        new = encode_native_organ_observations(self.model, self.variables, changed)
        kidneys = tuple(CANONICAL_ORGAN_VARIABLES).index('kidneys')
        torch.testing.assert_close(base[1][:, kidneys], new[1][:, kidneys], rtol=0, atol=0)

    def test_missing_is_not_a_zero_observation(self):
        self.tokens[:, 3:6] = 0
        self.tokens[:, :3] = float('nan')
        body, organs, presence = encode_native_organ_observations(self.model, self.variables, self.tokens)
        self.assertEqual(float(body.abs().sum()), 0)
        self.assertEqual(float(organs.abs().sum()), 0)
        self.assertEqual(float(presence.sum()), 0)

    def test_reject_uncontracted_treatment(self):
        self.tokens[:, -2] = 1
        with self.assertRaises(ValueError):
            encode_native_organ_observations(self.model, self.variables, self.tokens)
