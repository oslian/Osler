import unittest
import torch
from scripts.audit_optimizer_exposure_clock import matched_decay


class DecayClockTest(unittest.TestCase):
    def test_real_adamw_zero_grad_decay(self):
        p = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
        q = torch.nn.Parameter(p.detach().clone())
        a = torch.optim.AdamW([p], lr=1e-4, weight_decay=.01)
        b = torch.optim.AdamW([q], lr=.002, weight_decay=matched_decay(1e-4, .01, 528, .002))
        for _ in range(528):
            p.grad = torch.zeros_like(p); a.step()
        q.grad = torch.zeros_like(q); b.step()
        torch.testing.assert_close(p, q, rtol=0, atol=1e-12)

    def test_no_decay_and_invalid(self):
        self.assertEqual(matched_decay(.01, 0, 528, .1), 0)
        with self.assertRaises(ValueError):
            matched_decay(1, 1, 528, .1)
