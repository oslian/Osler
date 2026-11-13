import copy
import unittest

import numpy as np
import pandas as pd

from scripts.diagnose_screen_failures import (
    ALPHA, ARMS, CONTROLS, classify, scalar_paired, scalar_tables, summarize, verify_entry,
)


def entry():
    paired = {level: dict(n=n, mean=-1., p_less=0., ci95=[-1., -1.])
              for level, n in (('patient', 30), ('component', 3))}
    return dict(status='scored', patients=30, components=3, occurrences=30,
                normalized_mae={level: {arm: 0. if arm == ARMS[0] else 1. for arm in ARMS}
                                for level in ('patient', 'component')},
                comparisons={c: dict(paired=copy.deepcopy(paired), screen_pass=True) for c in CONTROLS})


class ScreenFailureTests(unittest.TestCase):
    def test_clear_gain_passes_unchanged_gate(self):
        result = classify(entry())
        self.assertEqual(result['first_unmet'], 'passed')
        self.assertEqual(summarize({'x': result})['full_screen'], ['x'])

    def test_disjoint_categories_and_overlap(self):
        e = entry()
        e['patients'] = 29
        for c in e['comparisons'].values():
            c['screen_pass'] = False
            c['paired']['patient'].update(n=29, mean=.1, p_less=.8, ci95=[-.1, .3])
        r = classify(e)
        self.assertEqual(r['first_unmet'], 'insufficient_support')
        self.assertEqual(set(r['all_failed']), {'support', 'mean', 'interval', 'nominal_test', 'family_test'})
        self.assertEqual(classify(dict(status='unsupported'))['first_unmet'], 'unsupported')

    def test_exact_boundaries(self):
        for field, value, category in (
            ('mean', 0., 'mean_not_better'),
            ('ci95', [-1., 0.], 'interval_inconclusive'),
            ('p_less', .05, 'nominal_test_inconclusive'),
            ('p_less', ALPHA, 'multiplicity_only'),
        ):
            e = entry(); comparison = e['comparisons'][CONTROLS[0]]
            comparison['paired']['patient'][field] = value
            if field == 'mean':
                comparison['paired']['patient'].update(p_less=.5, ci95=[-1., 1.])
            comparison['screen_pass'] = False
            self.assertEqual(classify(e)['first_unmet'], category)

    def test_malformed_and_tampered(self):
        e = entry(); del e['comparisons'][CONTROLS[0]]
        with self.assertRaises(ValueError): classify(e)
        e = entry(); e['comparisons'][CONTROLS[0]]['screen_pass'] = False
        with self.assertRaises(ValueError): classify(e)
        e = entry(); e['comparisons'][CONTROLS[0]]['paired']['patient']['mean'] = np.nan
        with self.assertRaises(ValueError): classify(e)

    def frame(self):
        return pd.DataFrame(dict(fold=0, patient=[str(i) for i in range(30)], variable='x',
            component=np.repeat(['a', 'b', 'c'], 10), truth=0.,
            **{arm: 0. if arm == ARMS[0] else 1. for arm in ARMS}))

    def test_scalar_identity_weighting_and_parity(self):
        f = self.frame()
        self.assertEqual(verify_entry(f, entry()), 0.)
        # A duplicate fold is another observation, not another patient.
        other = f.iloc[:1].copy(); other['fold'] = 1; other[ARMS[0]] = 2.
        people, groups = scalar_tables(pd.concat([f, other], ignore_index=True))
        self.assertEqual(len(people), 30)
        self.assertEqual(people[0][0], 1.)
        self.assertAlmostEqual(groups[0][0], .1)
        bad = pd.concat([f, f.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError): scalar_tables(bad)
        other['component'] = 'b'
        with self.assertRaises(ValueError): scalar_tables(pd.concat([f, other], ignore_index=True))

    def test_scalar_t_formula(self):
        from scipy.stats import ttest_1samp
        values = [-2., -.5, .1, -.8]
        mean, p = scalar_paired(values)
        self.assertAlmostEqual(mean, np.mean(values))
        self.assertAlmostEqual(p, ttest_1samp(values, 0, alternative='less').pvalue)
        self.assertEqual(scalar_paired([-1., -1.]), (-1., 0.))
        self.assertEqual(scalar_paired([0., 0.]), (0., 1.))
        self.assertEqual(scalar_paired([-1.]), (-1., None))


if __name__ == '__main__':
    unittest.main()
