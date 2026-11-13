import unittest
import torch
from scripts.audit_reconstruction_latest_coverage import latest_coverage


class LatestCoverageTests(unittest.TestCase):
    def test_latest_each_keeps_same_value_remeasurement_and_absolute_time(self):
        x = torch.zeros(5, 9)
        x[:, 3:6] = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        t = torch.tensor([1., 3., 7., 10., 20.])
        y, u = latest_coverage(x, t, 3, 1)
        torch.testing.assert_close(y, x[2:])
        torch.testing.assert_close(u, t[2:])
        torch.testing.assert_close(y[:, 3:6].any(0), x[:, 3:6].any(0))

    def test_suffix_and_missing_variables(self):
        x = torch.zeros(5, 9)
        x[0, 3] = 1
        x[1:, 4] = 1
        t = torch.arange(5.)
        y, u = latest_coverage(x, t, 3, 2)
        torch.testing.assert_close(y, x[[0, 3, 4]])
        torch.testing.assert_close(u, t[[0, 3, 4]])
        torch.testing.assert_close(latest_coverage(x, t, 3, 8)[0], x)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            latest_coverage(torch.zeros(1, 9), torch.zeros(1), 3, 0)
