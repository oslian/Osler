from copy import deepcopy
import unittest

import torch

from scripts.organ_assimilation import CHANNELS, OrganAssimilation, replay_assimilated
from scripts.train_content_retention import replay
from scripts.train_organ_assimilation import AssimilatedReconstruction, NAMES, assess, fit_arm, predict, validate_identities
from whole_body_jepa.physiology_state import PersistentPhysiologyStateLayer


class OrganAssimilationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.layer = PersistentPhysiologyStateLayer(body_dim=8, organ_dim=4,
            organ_count=3, adjacency=torch.eye(3)).double()
        self.bridge = OrganAssimilation(self.layer)
        self.example = dict(body=torch.randn(3, 8, dtype=torch.double),
            organs=torch.randn(3, 3, 4, dtype=torch.double),
            presence=torch.tensor([[1., 0., 1.], [1., 1., 0.], [0., 1., 1.]], dtype=torch.double),
            times=torch.tensor([5., 6., 8.], dtype=torch.double), anchor=8.)

    def test_zero_initialization_preserves_old_trajectory_and_clock(self):
        for channel in CHANNELS:
            for anchor in (8., 9.5):
                e = dict(self.example, anchor=anchor)
                actual = replay_assimilated(self.layer, self.bridge, e, channel)
                torch.testing.assert_close(actual.body, replay(self.layer, e), atol=0, rtol=0)
                self.assertEqual(actual.elapsed_hours.item(), anchor - 5.)
        matrix = self.bridge.body_expansion
        torch.testing.assert_close(matrix @ matrix.T, torch.eye(8, dtype=torch.double))

    def test_last_organ_reaches_body_only_with_immediate_channel(self):
        with torch.no_grad():
            self.bridge.projection.weight.fill_(.1)
        changed = dict(self.example, organs=self.example['organs'].clone())
        changed['organs'][-1, 1] += 1.
        for channel in CHANNELS:
            first = replay_assimilated(self.layer, self.bridge, self.example, channel)
            second = replay_assimilated(self.layer, self.bridge, changed, channel)
            difference = float((first.body - second.body).detach().abs().max())
            self.assertGreater(difference, 0.) if channel == 'immediate' else self.assertEqual(difference, 0.)
            self.assertEqual(second.elapsed_hours.item(), 3.)
        organs = self.example['organs'].clone().requires_grad_(True)
        output = replay_assimilated(self.layer, self.bridge, dict(self.example, organs=organs), 'immediate').body.sum()
        gradient, = torch.autograd.grad(output, organs)
        self.assertGreater(float(gradient[-1, 1].abs().max()), 0.)
        self.assertEqual(float(gradient[-1, 0].abs().max()), 0.)

    def test_pending_is_not_patient_memory_and_missing_does_not_flush(self):
        with torch.no_grad():
            self.bridge.projection.weight.fill_(.1)
        before = replay_assimilated(self.layer, self.bridge, self.example, 'delayed').body
        other = dict(self.example, organs=self.example['organs'] + 50)
        replay_assimilated(self.layer, self.bridge, other, 'delayed')
        after = replay_assimilated(self.layer, self.bridge, self.example, 'delayed').body
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        layer = self.layer
        state = layer.initialize(torch.ones(1, 8, dtype=torch.double),
            torch.ones(1, 3, 4, dtype=torch.double), torch.ones(1, 3, dtype=torch.double))
        absent = torch.zeros(1, 3, dtype=torch.double)
        updated = layer.update(state, torch.full_like(state.body, float('nan')),
            torch.full_like(state.organs, float('nan')), absent, state.elapsed_hours)
        pending = torch.ones(1, 12, dtype=torch.double)
        for channel in CHANNELS:
            actual, retained = self.bridge.correct(state, updated, pending, absent, channel)
            torch.testing.assert_close(actual.body, state.body, rtol=0, atol=0)
            torch.testing.assert_close(retained, pending, rtol=0, atol=0)

    def test_zero_start_is_trainable_and_checkpoint_replays(self):
        output = replay_assimilated(self.layer, self.bridge, self.example, 'immediate').body
        output.square().sum().backward()
        self.assertGreater(float(self.bridge.projection.weight.grad.abs().max()), 0.)
        optimizer = torch.optim.AdamW(self.bridge.parameters(), lr=1e-3)
        optimizer.step()
        other = OrganAssimilation(self.layer)
        other.load_state_dict(deepcopy(self.bridge.state_dict()), strict=True)
        torch.testing.assert_close(replay_assimilated(self.layer, other, self.example, 'immediate').body,
            replay_assimilated(self.layer, self.bridge, self.example, 'immediate').body, rtol=0, atol=0)

    def test_invalid_times_and_channels_rejected(self):
        for e in (dict(self.example, anchor=7.), dict(self.example, anchor=float('nan')),
                  dict(self.example, times=torch.tensor([5., 5., 8.], dtype=torch.double)),
                  dict(self.example, presence=self.example['presence'][:1])):
            with self.assertRaises(ValueError): replay_assimilated(self.layer, self.bridge, e, 'immediate')
        with self.assertRaises(ValueError): replay_assimilated(self.layer, self.bridge, self.example, 'typo')

    def test_single_event_zero_is_observed_and_delay_does_not_flush(self):
        with torch.no_grad():
            self.bridge.projection.weight.fill_(1.)
        state = self.layer.initialize(torch.ones(1, 8, dtype=torch.double),
            torch.ones(1, 3, 4, dtype=torch.double), torch.ones(1, 3, dtype=torch.double))
        pending = torch.zeros(1, 12, dtype=torch.double)
        present = torch.ones(1, 3, dtype=torch.double)
        updated = self.layer.update(state, state.body, torch.zeros_like(state.organs), present, state.elapsed_hours)
        actual, next_pending = self.bridge.correct(state, updated, pending, present, 'immediate')
        expected = updated.body + self.bridge.limit * torch.tanh(self.bridge.projection((updated.organs-state.organs).flatten(1)))
        torch.testing.assert_close(actual.body, expected, atol=0, rtol=0)
        self.assertGreater(float((actual.body-updated.body).detach().abs().max()), 0.)
        self.assertLessEqual(float((actual.body-updated.body).detach().abs().max()), self.bridge.limit)
        delayed, retained = self.bridge.correct(state, updated, pending, present, 'delayed')
        torch.testing.assert_close(delayed.body, updated.body, atol=0, rtol=0)
        torch.testing.assert_close(retained, next_pending, atol=0, rtol=0)
        # A second real observation consumes the preceding innovation exactly once.
        second = self.layer.update(updated, updated.body, updated.organs, present, state.elapsed_hours)
        final, empty = self.bridge.correct(updated, second, retained, present, 'delayed')
        torch.testing.assert_close(final.body, second.body + self.bridge.limit * torch.tanh(self.bridge.projection(retained)))
        self.assertEqual(float(empty.detach().abs().max()), 0.)

    def test_matched_training_and_scoring_contract(self):
        import pandas as pd
        from tests.test_organ_binding import OrganBindingTest
        from scripts.verify_organ_assimilation import direct_prediction
        setup = OrganBindingTest(); setup.setUp()
        e, bound = setup.example()
        counts = []
        for channel in CHANNELS:
            trained, log = fit_arm(setup.model, [e], [bound], channel)
            self.assertEqual([s['steps'] for s in log], [1, 1])
            counts.append(sum(p.numel() for p in trained.parameters()))
            fresh = AssimilatedReconstruction(setup.model, channel).eval()
            fresh.load_state_dict(trained.state_dict(), strict=True)
            first, diagnostics = predict(trained, [e], [bound])
            second, repeated = predict(fresh, [e], [bound])
            self.assertEqual(first.tolist(), second.tolist())
            self.assertEqual(diagnostics, repeated)
            with torch.no_grad():
                self.assertAlmostEqual(direct_prediction(fresh, e, bound), first[0], places=12)
            self.assertGreater(diagnostics['coordinates'], 0)
            for k, v in setup.model.trajectory.jepa.state_dict().items():
                torch.testing.assert_close(trained.model.trajectory.jepa.state_dict()[k], v, atol=0, rtol=0)
        self.assertEqual(len(set(counts)), 1)
        frame = pd.DataFrame(dict(patient=['a', 'a', 'b'], fold=[0, 1, 0], component=['x', 'x', 'y'],
                                  context=[2]*3, variable=['glucose']*3, truth=[0.]*3, **{n: [1., 3., 6.] for n in NAMES}))
        result = assess(frame, 9, 20)
        self.assertEqual(result['normalized_mae']['patient']['immediate'], 4.)
        self.assertFalse(any(c['screen_pass'] for c in result['comparisons'].values()))
        with self.assertRaises(ValueError): assess(pd.concat([frame, frame.iloc[:1]]), 9, 20)
        with self.assertRaises(ValueError): fit_arm(setup.model, [e], [], 'immediate')

    def test_null_identity_cannot_hide_a_harmed_patient(self):
        import pandas as pd
        frame = pd.DataFrame(dict(patient=[f'p{i}' for i in range(31)],
            component=[f'c{i%3}' for i in range(31)], variable=['glucose']*31,
            fold=[0]*31, context=[2]*31, truth=[0.]*31,
            **{n: ([0.]*30+[1000.] if n == 'immediate' else [1.]*31) for n in NAMES}))
        validate_identities(frame)
        for key in ('patient', 'component', 'variable', 'fold', 'context'):
            damaged = frame.copy(); damaged[key] = damaged[key].astype(object)
            damaged.loc[30, key] = None
            with self.assertRaises(ValueError): assess(damaged, 9, 20)
        for key, value in (('patient', ''), ('component', ''), ('context', 53),
                           ('fold', -1), ('context', 3)):
            damaged = frame.copy(); damaged.loc[30, key] = value
            with self.assertRaises(ValueError): validate_identities(damaged)

    def test_independent_schedule_handles_missing_and_single_events(self):
        from types import SimpleNamespace
        from torch import nn
        from scripts.verify_organ_assimilation import direct_prediction
        from scripts.train_organ_binding import select_arm
        candidate = SimpleNamespace(model=SimpleNamespace(trajectory=SimpleNamespace(layer=self.layer),
            readout=nn.Linear(8, 2).double()), bridge=self.bridge)
        with torch.no_grad(): self.bridge.projection.weight.fill_(.1)
        missing = deepcopy(self.example)
        missing['presence'][1] = 0
        missing['body'][1] = float('nan')
        missing['organs'][1] = 0
        single = {key: value[-1:] for key, value in self.example.items() if key != 'anchor'}
        single['anchor'] = 8.
        for e in (missing, single, dict(single, anchor=9.5)):
            e = dict(e, target=0)
            bound = torch.nn.functional.silu(e['organs'])
            for channel in CHANNELS:
                candidate.channel = channel
                with torch.no_grad():
                    state = replay_assimilated(self.layer, self.bridge, select_arm(e, bound, 'binding'), channel)
                    expected = float(candidate.model.readout(state.body)[0, 0])
                    self.assertAlmostEqual(direct_prediction(candidate, e, bound), expected, places=12)


if __name__ == '__main__':
    unittest.main()
