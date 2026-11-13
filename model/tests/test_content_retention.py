import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.train_content_retention import replay, shuffle_observed, content_loss, assess, fit_arm, predict_rows
from scripts.verify_content_retention import verify_predictions
from tests.test_native_body_reconstruction import BodyReconstructionTest
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations


class ContentRetentionTests(unittest.TestCase):
    setUp = BodyReconstructionTest.setUp

    def example(self, anchor=3.):
        variables = self.model.trajectory.variables
        clean, times = mask_native_target(self.tokens, self.times, variables, 'map')
        with torch.no_grad():
            body, organs, presence = encode_native_organ_observations(self.model.trajectory.jepa, variables, clean)
            frozen = float(self.model(self.tokens, self.times, anchor, 'map')[0])
        return dict(body=body, organs=organs, presence=presence, times=times,
            anchor=anchor, target=0, truth=1., mask=torch.tensor([False, True, True]),
            values=torch.zeros(3), patient='a', component='x', frozen=frozen)

    def test_cached_replay_matches_original_and_keeps_gradients(self):
        for anchor in (1., 3.):
            e = self.example(anchor)
            pred = self.model.readout(replay(self.model.trajectory.layer, e))[0, 0]
            torch.testing.assert_close(pred, pred.new_tensor(e['frozen']), atol=0, rtol=0)
            pred.backward()
            self.assertTrue(any(p.grad is not None for p in self.model.trajectory.layer.parameters()))

    def test_invalid_cached_times_rejected(self):
        e = self.example()
        for times in (torch.tensor([0., 0.]), torch.tensor([0., 4.]), torch.tensor([0., float('nan')])):
            with self.assertRaises(ValueError): replay(self.model.trajectory.layer, dict(e, times=times))

    def test_shuffle_preserves_observed_zero_and_excludes_singletons(self):
        values = torch.tensor([[0., 1., 99.], [2., 99., 99.], [3., 99., 99.]])
        mask = torch.tensor([[True, True, False], [True, False, False], [True, False, False]])
        shuffled, allowed, eligible = shuffle_observed(values, mask)
        torch.testing.assert_close(shuffled[:, 0], torch.tensor([3., 0., 2.]))
        self.assertEqual(eligible.tolist(), [True, False, False])
        self.assertEqual(int(allowed.sum()), 3)
        np.testing.assert_array_equal(sorted(shuffled[:, 0].tolist()), sorted(values[:, 0].tolist()))

    def test_detached_control_trains_head_not_state_and_ignores_missing(self):
        for detached in (False, True):
            state = torch.ones(1, 4, requires_grad=True)
            head = nn.Linear(4, 3)
            values = torch.tensor([0., float('nan'), 1.])
            loss = content_loss(head, state, values, torch.tensor([True, False, True]), detached)
            loss.backward()
            self.assertEqual(state.grad is None, detached)
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.)

    def test_training_does_not_change_encoder_or_original(self):
        original = {k: v.clone() for k, v in self.model.state_dict().items()}
        e = self.example()
        model, head, eligible, log = fit_arm(self.model, [e, dict(e, patient='b')], 'real')
        for key, value in original.items(): torch.testing.assert_close(self.model.state_dict()[key], value, atol=0, rtol=0)
        for key, value in self.model.trajectory.jepa.state_dict().items():
            torch.testing.assert_close(model.trajectory.jepa.state_dict()[key], value, atol=0, rtol=0)
        self.assertEqual(len(log), 2)
        rows = predict_rows(model, head, [e], eligible, 0, self.model.trajectory.variables)
        self.assertTrue(np.isfinite(rows.prediction).all())
        self.assertEqual(rows[rows.task == 'content'].variable.tolist(), ['creatinine', 'glucose'])
        self.assertTrue(rows.loc[rows.task == 'content', 'frozen'].isna().all())

    def test_patient_equal_scoring_and_contracts(self):
        frame = pd.DataFrame(dict(patient=['a', 'a', 'b'], fold=[0, 1, 0], component=['x', 'x', 'y'],
            variable=['glucose']*3, task=['masked']*3, truth=[0.]*3))
        for arm in ('primary', 'real', 'shuffled', 'frozen'): frame[arm] = [1., 3., 6.]
        result = assess(frame, 1, 20)
        self.assertEqual(result['normalized_mae']['patient']['real'], 4.)
        self.assertFalse(any(r['screen_pass'] for r in result['comparisons'].values()))
        with self.assertRaises(ValueError): assess(pd.concat([frame, frame.iloc[:1]]), 1, 20)
        bad = frame.copy(); bad.loc[0, 'real'] = float('nan')
        with self.assertRaises(ValueError): assess(bad, 1, 20)
        bad = frame.copy(); bad.loc[0, 'component'] = 'z'
        with self.assertRaises(ValueError): assess(bad, 1, 20)

    def test_verifier_rejects_identity_and_prediction_changes(self):
        frame = pd.DataFrame(dict(patient=['a'], component=['x'], fold=[0], context=[0],
            variable=['map'], task=['masked'], truth=[1.], frozen=[0.], real=[.5]))
        pred = frame.rename(columns={'real': 'prediction'})
        self.assertEqual(verify_predictions(frame, pred, 'real'), 1)
        bad = pred.copy(); bad.loc[0, 'prediction'] = .6
        with self.assertRaises(AssertionError): verify_predictions(frame, bad, 'real')
        bad = pred.copy(); bad.loc[0, 'patient'] = 'b'
        with self.assertRaises(AssertionError): verify_predictions(frame, bad, 'real')


if __name__ == '__main__': unittest.main()
