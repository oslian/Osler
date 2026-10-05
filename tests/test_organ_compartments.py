import unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch

from scripts.probe_organ_compartments import ARMS, replay_full, donor_indices, features, assess
from scripts.train_content_retention import replay
from scripts.score_organ_compartments import joint_passes, summarize
from tests import test_content_retention as content_tests


class OrganCompartmentTests(unittest.TestCase):
    setUp = content_tests.ContentRetentionTests.setUp
    example = content_tests.ContentRetentionTests.example

    def test_full_state_matches_native_path_and_organs(self):
        for anchor in (1., 3.):
            e = self.example(anchor)
            layer = self.model.trajectory.layer
            state = replay_full(layer, e)
            torch.testing.assert_close(state.body, replay(layer, e), atol=0, rtol=0)
            point = self.model.readout(state.body)[0, 0]
            self.assertEqual(float(point.detach()), e['frozen'])
            zero = layer.initialize(torch.zeros_like(state.body), torch.zeros_like(state.organs),
                                    torch.zeros_like(state.uncertainty))
            times = (e['times']-e['times'][0]).to(e['body'].dtype)
            direct = layer.replay_observations(zero, e['body'][None], e['organs'][None],
                                              e['presence'][None], times[None])
            direct = layer.advance(direct, e['body'].new_tensor([anchor-float(e['times'][-1])]))
            torch.testing.assert_close(state.organs, direct.organs, atol=0, rtol=0)
            e['truth'] = float('nan')
            torch.testing.assert_close(replay_full(layer, e).organs, state.organs, atol=0, rtol=0)

    def test_invalid_time_rejected(self):
        e = self.example()
        for times in (torch.tensor([0., 0.]), torch.tensor([0., 4.]), torch.tensor([0., float('nan')])):
            with self.assertRaises(ValueError): replay_full(self.model.trajectory.layer, dict(e, times=times))

    def test_donors_are_distinct_same_target_and_do_not_fill_singletons(self):
        ids, targets = np.array(['a', 'b', 'c', 'd', 'e']), np.array([0, 1, 0, 1, 2])
        d = donor_indices(ids, targets)
        self.assertEqual(d[-1], -1)
        self.assertTrue((ids[d[:4]] != ids[:4]).all())
        np.testing.assert_array_equal(targets[d[:4]], targets[:4])
        self.assertEqual(sorted(d[:4]), [0, 1, 2, 3])
        with self.assertRaises(ValueError): donor_indices(['a', 'a'], [0, 0])

    def test_augmented_width_and_identical_process(self):
        rng = np.random.default_rng(7)
        data = dict(body=rng.normal(size=(4, 64)), organs=rng.normal(size=(4, 96)),
                    raw=rng.normal(size=(4, 265)), target=np.zeros(4, dtype=int),
                    donor=np.array([3, 0, 1, 2]))
        projection = rng.normal(size=(64, 96))/8
        rows = np.array([0, 2])
        result = {k: features(data, rows, k, projection) for k in ('body', 'organs', 'expanded', 'shuffled', 'raw')}
        for k in ('organs', 'expanded', 'shuffled'):
            self.assertEqual(result[k].shape, (2, 319))
            np.testing.assert_array_equal(result[k][:, :64], data['body'][rows])
            np.testing.assert_array_equal(result[k][:, -159:], data['raw'][rows, 106:])
        np.testing.assert_array_equal(result['shuffled'][:, 64:160], data['organs'][data['donor'][rows]])
        np.testing.assert_allclose(result['expanded'][:, 64:160], data['body'][rows] @ projection)
        data['donor'][0] = -1
        with self.assertRaises(ValueError): features(data, rows, 'shuffled', projection)

    def test_paired_assessment_averages_repeated_patients(self):
        frame = pd.DataFrame(dict(patient=['a', 'a', 'b'], fold=[0, 1, 0], component=['x', 'x', 'y'],
                                  variable=['map']*3, truth=[0.]*3))
        for arm in (*ARMS, 'native'): frame[arm] = [1., 3., 6.]
        result = assess(frame, 7, 20)
        self.assertEqual(result['normalized_mae']['patient']['hgb_organs'], 4.)
        self.assertFalse(any(r['screen_pass'] for r in result['comparisons'].values()))
        with self.assertRaises(ValueError): assess(pd.concat([frame, frame.iloc[:1]]), 7, 20)
        bad = frame.copy(); bad.loc[0, 'native'] = np.nan
        with self.assertRaises(ValueError): assess(bad, 7, 20)

    def test_joint_keys_include_reader_on_both_sides(self):
        screens = {f'{r}_organs_minus_{r}_{b}': ['map']
                   for r in ('ridge', 'hgb') for b in ('body', 'shuffled', 'expanded', 'raw')}
        self.assertEqual(joint_passes(screens, ['map', 'bun'], ('body', 'shuffled', 'expanded')), ['map'])
        screens['hgb_organs_minus_hgb_shuffled'] = []
        self.assertEqual(joint_passes(screens, ['map'], ('body', 'shuffled', 'expanded')), [])

    def test_summary_keeps_unsupported_and_requires_all_controls(self):
        from scripts.probe_organ_compartments import CONTRASTS
        frame = pd.DataFrame(dict(patient=['a'], variable=['map'], fold=[0]))
        result = dict(normalized_mae={l: {a: 1. for a in (*ARMS, 'native')}
                                     for l in ('patient', 'component')},
                      comparisons={a+'_minus_'+b: dict(screen_pass=True) for a, b in CONTRASTS})
        with patch('scripts.score_organ_compartments.assess', return_value=result):
            targets, summary = summarize(frame, ['map', 'bun'])
        self.assertEqual(targets['bun']['status'], 'insufficient_support')
        self.assertEqual(summary['joint_incremental'], ['map'])
        self.assertEqual(summary['joint_over_raw'], ['map'])


if __name__ == '__main__': unittest.main()
