import unittest

import numpy as np
import pandas as pd

from scripts.score_full_organ_readout import ARMS, CONTROLS, assess, summarize, verify_frame


class FullOrganScoreTest(unittest.TestCase):
    @staticmethod
    def frame(n=30):
        frame = pd.DataFrame(dict(patient=[f'p{i:03}' for i in range(n)],
            component=[f's{i%3}' for i in range(n)], variable='v0', fold=0, truth=0.))
        for arm in ARMS:
            frame[arm] = 0. if arm == 'organs' else 1.
        return frame

    def test_repeated_folds_are_not_independent_patients(self):
        frame = self.frame(3)
        extra = frame.iloc[[0]].copy(); extra['fold'] = 1; extra['organs'] = 2.
        result = assess(pd.concat([frame, extra]), 12, repetitions=100)
        self.assertEqual(result['patients'], 3)
        self.assertEqual(result['query_occurrences'], 4)
        self.assertAlmostEqual(result['normalized_mae']['patient']['organs'], 1/3)
        self.assertAlmostEqual(result['normalized_mae']['component']['organs'], 1/3)
        self.assertFalse(any(c['screen_pass'] for c in result['comparisons'].values()))

    def test_fixed_five_comparisons_and_full_inventory(self):
        targets, summary = summarize(self.frame(), [f'v{i}' for i in range(53)], repetitions=100)
        self.assertEqual(len(targets), 53)
        self.assertEqual(targets['v52'], dict(status='unsupported'))
        self.assertEqual(summary['joint_incremental'], ['v0'])
        self.assertEqual(summary['joint_all_controls'], ['v0'])
        self.assertEqual(set(summary['screens']), {'organs_minus_'+c for c in CONTROLS})
        self.assertTrue(all(c['screen_pass'] for c in targets['v0']['comparisons'].values()))

    def test_incremental_success_not_full_success(self):
        frame = self.frame(); frame['native'] = 0.; frame['raw'] = 0.
        _, summary = summarize(frame, [f'v{i}' for i in range(53)], repetitions=100)
        self.assertEqual(summary['joint_incremental'], ['v0'])
        self.assertEqual(summary['joint_all_controls'], [])

    def test_minimum_patients_and_components(self):
        frame = self.frame(29)
        result = assess(frame, 1, repetitions=100)
        self.assertFalse(any(c['screen_pass'] for c in result['comparisons'].values()))
        frame = self.frame(); frame['component'] = 'one'
        result = assess(frame, 1, repetitions=100)
        self.assertFalse(any(c['screen_pass'] for c in result['comparisons'].values()))

    def test_duplicate_or_changed_patient_component_rejected(self):
        frame = self.frame()
        with self.assertRaises(ValueError):
            assess(pd.concat([frame, frame]), 1, repetitions=100)
        extra = frame.iloc[[0]].copy(); extra['fold'] = 1; extra['component'] = 'other'
        with self.assertRaises(ValueError):
            assess(pd.concat([frame, extra]), 1, repetitions=100)
        frame.loc[0, 'organs'] = np.nan
        with self.assertRaises(ValueError):
            assess(frame, 1, repetitions=100)

    def test_component_equalization_after_patient_equalization(self):
        frame = self.frame(4)
        frame['component'] = ['s0', 's0', 's0', 's1']
        frame['organs'] = [0., 0., 0., 2.]
        result = assess(frame, 1, repetitions=100)
        self.assertEqual(result['normalized_mae']['patient']['organs'], .5)
        self.assertEqual(result['normalized_mae']['component']['organs'], 1.)

    def test_prediction_identity_truth_and_native_verified(self):
        frame = self.frame(3)
        ids = frame.patient.tolist(); components = dict(zip(ids, frame.component))
        query = dict(truth=np.zeros(3), prediction=np.ones(3))
        verify_frame(frame, ids, query, 'v0', 0, components)
        with self.assertRaises(AssertionError):
            verify_frame(frame.iloc[::-1], ids, query, 'v0', 0, components)
        changed = frame.copy(); changed.loc[0, 'truth'] = 2.
        with self.assertRaises(AssertionError):
            verify_frame(changed, ids, query, 'v0', 0, components)


if __name__ == '__main__':
    unittest.main()
