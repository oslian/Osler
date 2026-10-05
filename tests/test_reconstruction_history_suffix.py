import unittest
import torch
from scripts.audit_reconstruction_history_suffix import suffix


class HistorySuffixTests(unittest.TestCase):
    def test_preserves_actual_times_and_latest_event(self):
        x=torch.arange(8).reshape(4,2);t=torch.tensor([2.,4.,8.,11.],dtype=torch.float64)
        a,b=suffix(x,t,2)
        torch.testing.assert_close(a,x[2:]);torch.testing.assert_close(b,t[2:])
        torch.testing.assert_close(suffix(x,t,8)[0],x)

    def test_empty_or_zero_invalid(self):
        with self.assertRaises(ValueError):suffix(torch.zeros(1,2),torch.zeros(1),0)
