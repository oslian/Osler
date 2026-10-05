import unittest
import torch
from tests import test_native_body_reconstruction as fixture
from scripts.audit_bun_dynamics_branches import disabled_dynamics


class DynamicsBranchTests(unittest.TestCase):
    setUp = fixture.BodyReconstructionTest.setUp

    def test_restores_hooks_after_error(self):
        layer = self.model.trajectory.layer
        counts = [len(m._forward_hooks) for m in (layer.body_dynamics, layer.organ_dynamics)]
        with self.assertRaises(RuntimeError):
            with disabled_dynamics(layer, True, True):
                raise RuntimeError('expected')
        self.assertEqual(counts, [len(m._forward_hooks) for m in (layer.body_dynamics, layer.organ_dynamics)])

    def test_body_off_disconnects_organs_from_current_body_readout(self):
        with torch.inference_mode():
            outputs = []
            for organs in (False, True):
                with disabled_dynamics(self.model.trajectory.layer, True, organs):
                    outputs.append(self.model(self.tokens, self.times, 3., 'map'))
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
            original = self.model(self.tokens, self.times, 3., 'map')
            with disabled_dynamics(self.model.trajectory.layer):
                torch.testing.assert_close(self.model(self.tokens, self.times, 3., 'map'), original, rtol=0, atol=0)
