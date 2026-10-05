import numpy as np
import unittest

from scripts.audit_readout_reachability import audit_targets, matrix_audit


class ReadoutReachabilityTests(unittest.TestCase):
    def test_full_rank_shared_state_can_change_outputs_independently(self):
        w = np.array([[1., 2., 0.], [0., 1., 3.]])
        result = matrix_audit(w)
        self.assertTrue(result['full_row_rank'])
        self.assertLess(result['output_projector_error64'], 1e-12)
        self.assertLess(result['output_projector_error32'], 1e-6)


    def test_collinear_heads_are_not_independently_reachable(self):
        result = matrix_audit([[1., 2.], [2., 4.]])
        self.assertEqual(result['rank'], 1)
        self.assertFalse(result['full_row_rank'])
        self.assertIsNone(result['condition_number'])
        self.assertGreater(result['output_projector_error64'], .1)


    def test_zero_heads_do_not_invalidate_supported_rank(self):
        status = {'a': dict(comparisons={'raw': {}}, criteria={'support': True}),
                  'b': dict(comparisons={}, first_unmet='unsupported')}
        r = audit_targets([[1., 0.], [0., 0.]], [1., 4.], ['a', 'b'], status)
        self.assertEqual(r['zero_heads'], [dict(target='b', bias=4., scored=False, supported=False)])
        self.assertTrue(r['subsets']['supported']['matrix']['full_row_rank'])


    def test_all_zero_readout_is_finite_and_explicit(self):
        r = matrix_audit(np.zeros((2, 3)))
        self.assertEqual(r['rank'], 0)
        self.assertIsNone(r['condition_number'])
        self.assertEqual(r['output_projector_error64'], 1.)


    def test_invalid_matrix_rejected(self):
        for w in ([], [1., 2.], [[np.nan]], [[np.inf]]):
            with self.subTest(matrix=w), self.assertRaises(ValueError):
                matrix_audit(w)


    def test_inventory_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            audit_targets([[1.]], [0.], ['a'], {})

    def test_missing_support_is_not_silently_treated_as_unsupported(self):
        with self.assertRaises(ValueError):
            audit_targets([[1.]], [0.], ['a'], {'a': {'comparisons': {}}})
