import copy
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.audit_aligned_coordinate_numerics import check_mapping
from scripts.train_aligned_coordinates import fit_axes


class MappingNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        body = rng.normal(size=(80, 64))
        raw = rng.normal(size=(80, 265))
        raw[:, 7] = 49.
        with threadpool_limits(limits=1):
            cls.mapping = fit_axes(body, raw, np.ones(80), list(range(80)), ['a'] * 80, 7)

    def test_valid_mapping(self):
        result = check_mapping(self.mapping)
        self.assertEqual(result['constant_outputs'], 1)
        self.assertTrue(result['original_body_preserved'])

    def test_nonfinite_coefficients_rejected(self):
        mapping = copy.deepcopy(self.mapping)
        mapping['aligned_body'].coef_[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'coefficients'):
            check_mapping(mapping)

    def test_negative_variance_not_silently_accepted(self):
        mapping = copy.deepcopy(self.mapping)
        mapping['body_scaler'].var_[0] = -1.
        with self.assertRaisesRegex(ValueError, 'Negative variance'):
            check_mapping(mapping)

    def test_nonunit_control_rejected(self):
        mapping = copy.deepcopy(self.mapping)
        mapping['random_axes'][:, 0] *= 2
        with self.assertRaisesRegex(ValueError, 'unit basis'):
            check_mapping(mapping)


if __name__ == '__main__':
    unittest.main()
