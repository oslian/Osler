import unittest
import torch
from scripts.audit_bun_history_strata import support, bins


class HistoryStrataTests(unittest.TestCase):
    def test_uses_latest_actual_measurement_age(self):
        x = torch.zeros(3, 6)
        x[:, 2:4] = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        x[1, 5] = 2/24
        t = torch.tensor([1., 3., 9.], dtype=torch.float64)
        result = support(x, t, 10., 2)
        self.assertAlmostEqual(result['maximum_latest_age_hours'], 9., places=6)
        self.assertEqual(result['independent_variables'], 2)
        self.assertEqual(bins(result), dict(events='1-8', span='6-24h', latest_age='6h+'))

    def test_fixed_boundaries(self):
        self.assertEqual(bins(dict(events=8, span_hours=6, maximum_latest_age_hours=6)),
                         dict(events='1-8', span='0-6h', latest_age='0-6h'))
        self.assertEqual(bins(dict(events=32, span_hours=24, maximum_latest_age_hours=7))['events'], '9-32')

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            support(torch.empty(0, 6), torch.empty(0), 0., 2)
