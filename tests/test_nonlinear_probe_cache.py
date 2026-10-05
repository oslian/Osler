import unittest
import numpy as np
import torch

from scripts.cache_nonlinear_probe import decode_features, encode_features
from scripts.audit_reconstruction_raw_state import summarize


class ProbeCacheTests(unittest.TestCase):
    def test_lossless_features(self):
        values = dict(body=np.arange(64, dtype=np.float32), raw=np.arange(265, dtype=float), truth=1.25, prediction=-2.)
        payload, digest = encode_features(**values)
        restored = decode_features(payload, digest)
        for name, value in values.items():
            np.testing.assert_array_equal(restored[name], value)

    def test_corruption_rejected(self):
        payload, digest = encode_features(np.zeros(64), np.zeros(265), 1., 2.)
        with self.assertRaises(ValueError):
            decode_features(payload+b'x', digest)

    def test_wrong_dimension_or_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            encode_features(np.zeros(63), np.zeros(265), 1., 2.)
        with self.assertRaises(ValueError):
            encode_features(np.zeros(64), np.zeros(265), np.nan, 2.)

    def test_raw_summary_hides_target_and_siblings(self):
        variables = ['hemoglobin', 'hematocrit', 'glucose']
        tokens = torch.tensor([[1., 2., 3., 1., 1., 1., 0., 0., 0.]])
        raw, process = summarize(tokens, torch.tensor([0.]), 0., variables, 'hemoglobin')
        np.testing.assert_array_equal(raw[:6], [0., 0., 3., 0., 0., 3.])
        np.testing.assert_array_equal(raw[6:9], [0., 0., 1.])
        np.testing.assert_array_equal(process[:6], np.zeros(6))


if __name__ == '__main__':
    unittest.main()
