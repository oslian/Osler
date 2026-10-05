import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from scripts.score_additional_reconstruction import paired_summary, summarize_target, verify_frame


class ScoringTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(patient=[str(i) for i in range(30)], target='x',
            component=np.repeat(['a', 'b', 'c'], 10), fold=0, truth=10., fit_scale=2.,
            real=10., process=12., population=14.))

    def test_paired_and_normalized(self):
        r = summarize_target(self.frame(), 1, 100)
        self.assertTrue(r['descriptive_screen_pass'])
        self.assertEqual(r['raw_mae']['patient']['process'], 2)
        self.assertEqual(r['normalized_mae']['patient']['process'], 1)
        self.assertEqual(r['comparisons']['process']['component']['ci95'], [-2, -2])

    def test_component_equal_not_row_equal(self):
        f = self.frame().iloc[:21].copy()
        f.loc[f.component == 'c', 'real'] = 40
        r = summarize_target(f, 1, 100)
        self.assertAlmostEqual(r['raw_mae']['patient']['real'], 30 / 21)
        self.assertEqual(r['raw_mae']['component']['real'], 10)
        self.assertFalse(r['descriptive_screen_pass'])

    def test_invalid_inputs(self):
        f = self.frame()
        with self.assertRaises(ValueError): summarize_target(pd.concat([f, f]), 1, 100)
        f.loc[0, 'fit_scale'] = 0
        with self.assertRaises(ValueError): summarize_target(f, 1, 100)
        self.assertIsNone(paired_summary([-1], 1, 100)['p_less'])
        self.assertEqual(paired_summary([0, 0], 1, 100)['p_less'], 1)

    def test_exact_identity_and_label(self):
        f = self.frame()
        s = dict(people=f.patient.tolist(), variables=['x'], centers=np.full((30, 1), 10.))
        roles = dict.fromkeys(f.patient, 'test')
        verify_frame(f, s, roles, 0)
        with self.assertRaises(ValueError): verify_frame(f.iloc[:-1], s, roles, 0)
        f.loc[0, 'truth'] = 9
        with self.assertRaises(ValueError): verify_frame(f, s, roles, 0)

    def test_full_family_threshold(self):
        # Would pass the obsolete 39-target cutoff, but not this 47-target family.
        stat = dict(n=30, mean=-1., p_less=.00115, ci95=[-2., -.1])
        with patch('scripts.score_additional_reconstruction.paired_summary', return_value=stat):
            self.assertFalse(summarize_target(self.frame(), 1, 100)['descriptive_screen_pass'])


if __name__ == '__main__': unittest.main()
