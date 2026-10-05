import unittest
import torch
from osler_jepa.patient_state import NeuralBeliefFilter
from whole_body_jepa.native_predict_only_control import PredictOnlyControl
from whole_body_jepa.native_projection_control import target_projection_input


class PredictOnlyTest(unittest.TestCase):
    def test_extra_events_do_not_invent_correction(self):
        torch.manual_seed(17)
        model = PredictOnlyControl(19, 4, 3, 1, 0)
        reference = NeuralBeliefFilter(19, 4)
        reference.load_state_dict(model.state_dict())
        x = torch.randn(2, 5, 19)
        x[:, :, 3:6] = 0; x[:, :, -2] = 0
        x[:, [0, 4], 4] = 1; x[:, 1:4, 3] = 1
        x[:, :, -1] = torch.tensor([0., .02, .04, .07, .1])
        mask = x[:, :, 3:6].bool().any(-1)
        actual = model(x, mask)
        expected = reference(target_projection_input(x, 3, 1, 0), x[:, :, 4].bool())
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_only_time_after_last_measurement_decays_state(self):
        torch.manual_seed(19)
        model = PredictOnlyControl(19, 4, 3, 1, 0)
        x = torch.zeros(1, 2, 19); x[:, 0, 4] = 1; x[:, 1, -2] = 1
        x[:, 1, -1] = .1
        start = model(x[:, :1], torch.ones(1, 1, dtype=torch.bool))[0]
        end = model(x, torch.tensor([[True, False]]))[0]
        expected = start*torch.exp(-torch.nn.functional.softplus(model.decay_rate)*.1)
        torch.testing.assert_close(end, expected)

    def test_no_measurements_or_padding_has_zero_state(self):
        model = PredictOnlyControl(19, 4, 3, 1, 0)
        x = torch.randn(1, 3, 19); x[:, :, 3:6] = 0; x[:, :, -2] = 0
        result = model(x, torch.zeros(1, 3, dtype=torch.bool))[0]
        torch.testing.assert_close(result, torch.zeros_like(result))
