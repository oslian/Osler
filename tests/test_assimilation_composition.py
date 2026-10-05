import unittest

import pandas as pd

from scripts.diagnose_assimilation_composition import CONTROLS, contrast, diagnose, verify_arithmetic


def frame(prefix, groups):
    rows = []
    for component, values in groups.items():
        for v in values:
            rows.append(dict(patient=prefix+str(len(rows)), component=component, variable='x',
                context=0, fold=0, truth=0., immediate=v, **{c: 1. for c in CONTROLS}))
    return pd.DataFrame(rows)


class AssimilationCompositionTest(unittest.TestCase):
    def test_weight_only_reversal(self):
        a, b = frame('f', {'a': [0.]*9, 'b': [2.]}), frame('q', {'a': [0.], 'b': [2.]*9})
        r = contrast(a, b, 'unbridged'); verify_arithmetic(r)
        self.assertAlmostEqual(r['within'], 0)
        self.assertAlmostEqual(r['composition'], 1.6)
        self.assertLess(r['roles']['fit']['mean_delta'], 0)
        self.assertGreater(r['roles']['query']['mean_delta'], 0)

    def test_within_only(self):
        a, b = frame('f', {'a': [0.], 'b': [1.]}), frame('q', {'a': [1.], 'b': [2.]})
        r = contrast(a, b, 'unbridged'); verify_arithmetic(r)
        self.assertAlmostEqual(r['within'], 1)
        self.assertAlmostEqual(r['composition'], 0)
        r['within'] = 0.
        with self.assertRaises(ValueError): verify_arithmetic(r)

    def test_restriction_not_imputed(self):
        a, b = frame('f', {'a': [0.], 'b': [20.]}), frame('q', {'a': [1.]})
        r = contrast(a, b, 'unbridged'); verify_arithmetic(r)
        self.assertEqual(r['common_fit_fraction'], .5)
        self.assertEqual(r['common_gap'], 1.)
        self.assertNotEqual(r['all_source_gap'], r['common_gap'])
        self.assertEqual(contrast(a, frame('q', {'c': [1.]}), 'unbridged')['status'], 'no_common_source')
        self.assertEqual(contrast(a.iloc[:0], b, 'unbridged')['status'], 'missing_role_support')

    def test_inventory_and_invalid_role_contracts(self):
        a, b = frame('f', {'a': [0.]}), frame('q', {'a': [1.]})
        r = diagnose(a, b, ['x', 'y'])
        self.assertEqual(r['folds']['0']['y']['unbridged']['status'], 'missing_role_support')
        with self.assertRaises(ValueError): diagnose(a, a, ['x'])
        bad = b.copy(); bad.loc[0, 'immediate'] = float('nan')
        with self.assertRaises(ValueError): diagnose(a, bad, ['x'])
        bad = b.copy(); bad.loc[0, 'context'] = 1
        with self.assertRaises(ValueError): diagnose(a, bad, ['x'])

    def test_folds_cannot_manufacture_overlap(self):
        a0, b0 = frame('f0', {'a': [0.]}), frame('q0', {'b': [1.]})
        a1, b1 = frame('f1', {'b': [0.]}), frame('q1', {'a': [1.]})
        a1['fold'] = 1; b1['fold'] = 1
        r = diagnose(pd.concat([a0, a1]), pd.concat([b0, b1]), ['x'])
        self.assertEqual(r['summary']['unbridged']['status_counts'], {'no_common_source': 2})


if __name__ == '__main__':
    unittest.main()
