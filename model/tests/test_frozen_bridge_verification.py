from copy import deepcopy
import unittest

import pandas as pd

from scripts.train_frozen_bridge import assess, describe, SEED
from scripts.verify_frozen_bridge import compare, independent_results, diagnose
from tests import test_frozen_bridge as fixtures


class FrozenVerificationTest(unittest.TestCase):
    def test_patient_weight_and_statistics(self):
        f = fixtures.FrozenBridgeTest().frame()
        expected = describe(f, ['map', 'bun'])
        expected['map'] = assess(f, SEED, repetitions=20)
        actual = independent_results(f, ['map', 'bun'], selection=True, repetitions=20)
        self.assertLess(compare(expected, actual), 1e-12)
        self.assertEqual(actual['map']['normalized_mae']['patient']['fixed_immediate'], 2.)
        self.assertEqual(actual['bun'], dict(status='unsupported'))

    def test_tamper_rejected(self):
        f = fixtures.FrozenBridgeTest().frame()
        result = independent_results(f, ['map'], True, repetitions=20)
        for kind in ('mean', 'screen', 'support', 'missing'):
            bad = deepcopy(result)
            if kind == 'mean': bad['map']['normalized_mae']['patient']['fixed_immediate'] += .01
            elif kind == 'screen': bad['map']['comparisons']['raw']['screen_pass'] = True
            elif kind == 'support': bad['map']['patients'] += 1
            else: del bad['map']
            with self.assertRaises(ValueError): compare(bad, result)
        with self.assertRaises(ValueError): independent_results(f.drop(columns='raw'), ['map'], True)
        with self.assertRaises(ValueError): independent_results(f.assign(truth=float('nan')), ['map'])

    def test_descriptive_reversal(self):
        f = fixtures.FrozenBridgeTest().frame()
        fit = describe(f.assign(fixed_immediate=0.), ['map'])
        query = describe(f.assign(fixed_immediate=4.), ['map'])
        r = diagnose(dict(fit=fit, selection=query))
        self.assertEqual(r['selection']['contrasts']['bound_start']['fit_gain_query_non_gain'], 1)
        self.assertEqual(r['selection']['contrasts']['bound_start']['mean_gain'], 0)

    def test_supported_gain_and_insufficient_component_rejection(self):
        template = fixtures.FrozenBridgeTest().frame().iloc[0].to_dict()
        rows = []
        for i in range(60):
            r = dict(template, patient=f'p{i:03}', component=f's{i//12}', fold=0,
                     fixed_immediate=.01*(i%7))
            rows.append(r)
        f = pd.DataFrame(rows)
        expected = dict(unused=dict(status='unsupported'), map=assess(f, SEED+1000, repetitions=40))
        actual = independent_results(f, ['unused', 'map'], True, repetitions=40)
        self.assertLess(compare(expected, actual), 1e-12)
        self.assertTrue(all(c['screen_pass'] for c in actual['map']['comparisons'].values()))
        unsupported = independent_results(f.assign(component='only'), ['unused', 'map'], True, repetitions=40)
        self.assertTrue(all(not c['screen_pass'] for c in unsupported['map']['comparisons'].values()))


if __name__ == '__main__': unittest.main()
