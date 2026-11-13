import unittest
import numpy as np
from scripts.audit_gradient_sample_size import split_prefixes, cosine


class BatchProbeTest(unittest.TestCase):
    def test_disjoint_and_capped(self):
        for size in (17, 63, 128, 200):
            a, b = split_prefixes(size, np.random.default_rng(19))
            self.assertFalse(set(a) & set(b))
            self.assertEqual(len(a), min(size // 2, 64))
            self.assertEqual(len(a), len(b))

    def test_undefined_zero_is_not_alignment(self):
        self.assertIsNone(cosine(np.zeros(2), np.ones(2)))
        self.assertAlmostEqual(cosine(np.ones(2), -np.ones(2)), -1.)
