import unittest

import numpy as np
import pandas as pd

from scripts.summarize_continuation_selection import change_agreement, summarize_fold


class SourceSummaryTests(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame(dict(patient=['a', 'b', 'c'], target='x', fold=0,
            component=['s', 's', 't'], real=[1., 1., 4.], process=[2., 2., 3.],
            population=[2., 2., 3.], truth=0., fit_scale=1.))
        self.fresh = self.base[['patient', 'target', 'fold']].assign(real10=[0., 0., 5.], process10=3.)

    def test_patient_and_component_weighting_remain_distinct(self):
        result = summarize_fold(self.base, self.fresh)['x']
        self.assertAlmostEqual(result['normalized_mae']['patient']['real10'], 5/3)
        self.assertAlmostEqual(result['normalized_mae']['component']['real10'], 2.5)
        self.assertEqual(result['component_harms']['population']['recurring'], 1)
        self.assertEqual(result['component_harms']['population']['supported_recurring'], 0)
        self.assertEqual(len(result['per_component']), 2)

    def test_matched_identity_required(self):
        with self.assertRaises(AssertionError):
            summarize_fold(self.base, self.fresh.iloc[::-1])

    def test_invalid_scale_rejected(self):
        self.base.loc[0, 'fit_scale'] = 0
        with self.assertRaises(ValueError):
            summarize_fold(self.base, self.fresh)

    def test_missing_outer_support_not_counted_as_disagreement(self):
        result = change_agreement([2]*5, [1]*5, [2]*5, [3]*5,
                                  [2, 2, 2, 2, np.nan], [1, 3, 1, 3, np.nan])
        self.assertEqual(result['full_monitor_sign_disagreements'], 5)
        self.assertEqual(result['full_outer_sign_disagreements'], 2)
        self.assertEqual(result['supported_outer_comparisons'], 4)
        self.assertIsNone(result['outer_real_change'][4])

    def test_missing_source_evidence_rejected(self):
        with self.assertRaises(ValueError):
            change_agreement([np.nan]*5, *[[1]*5]*5)


if __name__ == '__main__':
    unittest.main()
