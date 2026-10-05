import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.score_monitored_continuation_outer import (
    verify_bound_files, verify_prediction_pair,
)
from whole_body_jepa.multisource_intervals import sha256


class ContinuationOuterContractTests(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame(dict(patient=['a', 'b'], target=['x', 'x'],
                                      fold=[2, 2], component=['c1', 'c2']))
        self.fresh = self.base[['patient', 'target', 'fold']].assign(
            real10=[1., 2.], process10=[2., 3.])
        self.meta = dict(fold=2, source_fold=0, role='test', examples=2, patients=2,
                         candidate_only=True, promotion_allowed=False)
        self.components = {'a': 'c1', 'b': 'c2'}

    def test_certified_reuse_is_accepted_but_wrong_fit_is_not(self):
        verify_prediction_pair(self.base, self.fresh, self.meta, 2, self.components)
        self.meta['source_fold'] = 2
        with self.assertRaisesRegex(ValueError, 'fitted fold'):
            verify_prediction_pair(self.base, self.fresh, self.meta, 2, self.components)

    def test_misaligned_rows_are_not_silently_scored(self):
        fresh = self.fresh.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(AssertionError):
            verify_prediction_pair(self.base, fresh, self.meta, 2, self.components)

    def test_changed_group_provenance_is_rejected(self):
        self.components['a'] = 'c2'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            verify_prediction_pair(self.base, self.fresh, self.meta, 2, self.components)

    def test_dependency_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'dependency.txt'
            path.write_text('original')
            bindings = {str(path): sha256(path)}
            verify_bound_files(bindings, {})
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'dependency'):
                verify_bound_files(bindings, {})

    def test_nonfinite_prediction_is_rejected(self):
        self.fresh.loc[0, 'real10'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            verify_prediction_pair(self.base, self.fresh, self.meta, 2, self.components)


if __name__ == '__main__':
    unittest.main()
