from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from scripts.train_pair_gradient import describe, assess, CONTROLS, SEED
from scripts.verify_pair_gradient import scalar_results, training_summary
from scripts.verify_frozen_bridge import compare


class PairVerificationTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(patient=str(i), component=str(i%3), fold=f, context=0,
            variable='map', truth=float(i%5), pcgrad=float(i%5)+.1,
            **{c:float(i%5)+.3+j*.01 for j,c in enumerate(CONTROLS)})
            for i in range(33) for f in range(1+i%2)])

    def test_independent_grouping_statistics(self):
        f = self.frame()
        self.assertLess(compare(describe(f, ['map', 'bun']), scalar_results(f, ['map', 'bun'])), 1e-12)
        expected = assess(f, SEED)
        actual = scalar_results(f, ['map', 'bun'], True)
        self.assertLess(compare(expected, actual['map']), 1e-12)
        self.assertTrue(all(r['screen_pass'] for r in actual['map']['comparisons'].values()))
        self.assertEqual(actual['bun'], dict(status='unsupported'))

    def test_bad_rows_and_small_support(self):
        f = self.frame()
        with self.assertRaises(ValueError):
            scalar_results(pd.concat([f, f.iloc[:1]]), ['map'])
        with self.assertRaises(ValueError):
            scalar_results(f.assign(pcgrad=np.nan), ['map'])
        with self.assertRaises(ValueError):
            scalar_results(f, ['bun'])
        r = scalar_results(f[f.patient.isin(['0', '1'])], ['map'], True)['map']
        self.assertFalse(any(x['screen_pass'] for x in r['comparisons'].values()))

    def test_budget_and_projection_metadata(self):
        rows = [dict(epoch=i, steps=1024, examples=2048, loss=.5, conflicting_pairs=300,
            projected_pairs=290, proposed_harmful_dots=5, actual_harmful_dots=700,
            maximum_proposed_harm=.1, maximum_actual_harm=.2) for i in (1, 2)]
        self.assertEqual(training_summary(rows, 'pcgrad')['steps'], 2048)
        with self.assertRaises(ValueError):
            training_summary(rows, 'mean')
        for field, value in [('steps', 1023), ('projected_pairs', 301), ('loss', float('nan'))]:
            bad = deepcopy(rows)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                training_summary(bad, 'pcgrad')


if __name__ == '__main__':
    unittest.main()
