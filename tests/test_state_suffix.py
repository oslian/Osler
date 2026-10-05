from dataclasses import replace
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.probe_state_suffix import suffix_states, assess, ARMS
from scripts.train_content_retention import replay
from scripts.verify_state_suffix import numeric_summary_parity, assessment_parity
from scripts.probe_assimilation_content import trace_content
from scripts.probe_observed_content import last_observed
from tests.test_native_body_reconstruction import BodyReconstructionTest
from whole_body_jepa.native_body_reconstruction import mask_native_target
from whole_body_jepa.native_organ_observations import encode_native_organ_observations
from whole_body_jepa.physiology_state import PersistentPhysiologyState


class ToyLayer:
    body_dim = organ_dim = organ_count = 1

    def initialize(self, body, organs, presence):
        return PersistentPhysiologyState(body, organs, 1-presence, body.new_zeros(len(body)))

    def advance(self, state, delta):
        return replace(state, body=state.body+state.organs[:, 0]*delta[:, None],
                       elapsed_hours=state.elapsed_hours+delta)

    def update(self, state, body, organs, presence, delta):
        return replace(state, body=.5*state.body+body+delta[:, None], organs=organs)


class SuffixTests(unittest.TestCase):
    def setUp(self):
        self.layer = ToyLayer()
        self.e = dict(body=torch.tensor([[2.], [7.], [11.]]),
            organs=torch.tensor([[[1.]], [[2.]], [[3.]]]), presence=torch.ones(3, 1),
            times=torch.tensor([10., 11., 13.], dtype=torch.float64), anchor=15.)

    def test_batched_paths_match_independent_replays(self):
        positions = np.array([0, 1, 2, -1, 0])
        r = suffix_states(self.layer, self.e, positions)
        torch.testing.assert_close(r['body'], replay(self.layer, self.e)[0], atol=0, rtol=0)
        for column, start in enumerate(positions):
            if start < 0:
                for arm in ('post', 'advance', 'update'):
                    self.assertEqual(r[arm][column].item(), 0.)
                continue
            for mode in ('post', 'advance', 'update', 'body'):
                state = self.layer.initialize(torch.zeros(1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 1))
                t = self.e['times']-self.e['times'][0]
                for i in range(3):
                    dt = (t[i:i+1].float()-state.elapsed_hours)
                    if i <= start or mode in ('advance', 'body'):
                        state = self.layer.advance(state, dt)
                    else:
                        state = replace(state, elapsed_hours=t[i:i+1].float())
                    if i <= start or mode in ('update', 'body'):
                        state = self.layer.update(state, self.e['body'][i:i+1], self.e['organs'][i:i+1],
                            self.e['presence'][i:i+1], dt)
                if mode in ('advance', 'body'): state = self.layer.advance(state, torch.tensor([2.]))
                actual = r[mode] if mode == 'body' else r[mode][column]
                torch.testing.assert_close(actual, state.body[0], atol=0, rtol=0)
        self.assertNotEqual(r['advance'][0].item(), r['update'][0].item())

    def test_later_values_do_not_affect_post_or_advance(self):
        positions = np.array([0])
        r = suffix_states(self.layer, self.e, positions)
        other = dict(self.e, body=self.e['body'].clone())
        other['body'][1:] += 100
        changed = suffix_states(self.layer, other, positions)
        for arm in ('post', 'advance'): torch.testing.assert_close(r[arm], changed[arm], atol=0, rtol=0)
        self.assertGreater(float((r['update']-changed['update']).abs().sum()), 0)

    def test_zero_is_known_and_no_suffix_collapses(self):
        e = dict(self.e, body=torch.zeros(1, 1), organs=torch.zeros(1, 1, 1),
                 presence=torch.ones(1, 1), times=torch.tensor([10.]), anchor=10.)
        r = suffix_states(self.layer, e, np.array([0, -1]))
        for arm in ('post', 'advance', 'update'): torch.testing.assert_close(r[arm][0], r['body'], atol=0, rtol=0)

    def test_invalid_histories_fail_closed(self):
        for times in (torch.tensor([10., 10., 13.]), torch.tensor([10., 11., 16.]),
                      torch.tensor([10., float('nan'), 13.])):
            with self.assertRaises(ValueError): suffix_states(self.layer, dict(self.e, times=times), np.array([0]))
        for p in (np.array([-2]), np.array([3]), np.array([.5])):
            with self.assertRaises(ValueError): suffix_states(self.layer, self.e, p)

    def test_patient_equal_scoring_and_duplicate_guard(self):
        frame = pd.DataFrame(dict(patient=['a', 'a', 'b'], fold=[0, 1, 0],
            component=['x', 'x', 'y'], variable=['glucose']*3, truth=[0.]*3))
        for arm in ARMS: frame[arm] = [1., 3., 6.]
        result = assess(frame, 1, repetitions=20)
        self.assertEqual(result['normalized_mae']['patient']['ridge_update'], 4.)
        self.assertEqual(len(result['comparisons']), 8)
        self.assertEqual(result['descriptive_interaction']['patient']['ridge'], 0.)
        self.assertFalse(any(x['screen_pass'] for x in result['comparisons'].values()))
        with self.assertRaises(ValueError): assess(pd.concat([frame, frame.iloc[:1]]), 1, repetitions=20)

    def test_summary_roundoff_cannot_change_gate_or_paired_statistics(self):
        tiny = {'a': 1.+np.finfo(float).eps}
        self.assertGreater(numeric_summary_parity(tiny, {'a': 1.}), 0.)
        for wrong in (1.000001, float('nan'), float('inf')):
            with self.assertRaises(AssertionError): numeric_summary_parity({'a': wrong}, {'a': 1.})
        r = dict(normalized_mae={'a': 1.}, comparisons={'pass': False, 'p': .1}, patients=40)
        expected = dict(targets={'v': r}, summary={'variables': 1})
        self.assertEqual(assessment_parity({'v': r}, expected['summary'], expected), 0.)
        for change in ({'pass': True, 'p': .1}, {'pass': False, 'p': .1+1e-15}):
            with self.assertRaises(AssertionError):
                assessment_parity({'v': dict(r, comparisons=change)}, expected['summary'], expected)


class NeuralSuffixTests(unittest.TestCase):
    setUp = BodyReconstructionTest.setUp

    def test_original_snapshots_and_no_later_update_parity(self):
        with torch.no_grad():
            variables = self.model.trajectory.variables
            tokens, times = mask_native_target(self.tokens, self.times, variables, 'map')
            body, organs, presence = encode_native_organ_observations(self.model.trajectory.jepa, variables, tokens)
            _, positions = last_observed(tokens, times, 3., len(variables))
            e = dict(body=body, organs=organs, presence=presence, times=times, anchor=3.)
            result = suffix_states(self.model.trajectory.layer, e, positions)
            original = trace_content(self.model, self.tokens, self.times, 3., 'map')
            torch.testing.assert_close(result['post'], original['post'], atol=0, rtol=0)
            torch.testing.assert_close(result['body'], original['final'], atol=0, rtol=0)
            for c in np.flatnonzero(positions == len(times)-1):
                torch.testing.assert_close(result['advance'][c], result['body'], atol=1e-6, rtol=0)
                torch.testing.assert_close(result['update'][c], result['post'][c], atol=0, rtol=0)


if __name__ == '__main__': unittest.main()
