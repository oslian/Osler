import unittest
import torch
from whole_body_jepa.native_projection_control import target_projection_input
from whole_body_jepa.native_event_encoder import NativeEventFilter
from osler_jepa.patient_state import NeuralBeliefFilter


class ProjectionControlTest(unittest.TestCase):
    def test_schema_and_gradient(self):
        x = torch.arange(19, dtype=torch.float32).reshape(1, 19).requires_grad_()
        y = target_projection_input(x, 3, 1, 0)
        for i in (1, 4, 7, 9, 18):
            self.assertEqual(y[0, i], x[0, i])
        self.assertEqual(y[0, 6], 7)
        self.assertEqual(y[0, 10], x[0, 4])
        self.assertTrue((y[0, 12:18] == 0).all())
        y.sum().backward()
        self.assertEqual(x.grad[0, 0], 0)
        self.assertEqual(x.grad[0, 1], 1)

    def test_activation_is_not_removed_with_projection_content(self):
        model = NativeEventFilter(19, 4)
        x = torch.zeros(1, 3, 19)
        x[0, 0, 3] = 1  # Non-target observation still activates this step.
        x[0, 1, 4] = 1
        x[0, 2, -2] = 1  # Treatment activation also remains.
        active = x[:, :, 3:6].bool().any(-1)
        hook = model.observation_projection.register_forward_pre_hook(
            lambda m, a: (target_projection_input(a[0], 3, 1, 0),))
        observed = []
        capture = model.observation_projection.register_forward_hook(
            lambda m, a, o: observed.append(a[0].detach().clone()))
        result, _ = model(x, active)
        reference = NeuralBeliefFilter(19, 4)
        reference.load_state_dict(model.state_dict())
        expected, _ = reference(target_projection_input(x, 3, 1, 0),
            active | x[:, :, -2].bool())
        torch.testing.assert_close(result, expected)
        self.assertEqual(len(observed), 3)
        self.assertEqual(observed[0][0, 3:6].sum(), 0)
        self.assertEqual(observed[2][0, -2], 0)
        self.assertTrue(torch.isfinite(result).all())
        capture.remove(); hook.remove()

    def test_invalid_schema(self):
        with self.assertRaises(ValueError):
            target_projection_input(torch.zeros(1, 19), 3, 1, 2)
