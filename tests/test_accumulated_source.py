from copy import deepcopy
import unittest
import torch
from scripts.train_accumulated_source import accumulated_step


class AccumulationTest(unittest.TestCase):
    def test_equals_explicit_batch_gradient_update(self):
        torch.manual_seed(1)
        model = torch.nn.Linear(2, 1).double()
        reference = deepcopy(model)
        x = torch.tensor([[1., 2.], [3., 4.], [-1., 0.]], dtype=torch.float64)
        y = torch.tensor([[1.], [2.], [0.]], dtype=torch.float64)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        refopt = torch.optim.AdamW(reference.parameters(), lr=1e-4)
        def loss(m, i):
            return torch.nn.functional.smooth_l1_loss(m(x[i:i+1]), y[i:i+1])
        accumulated_step(model, opt, list(range(3)), loss)
        refopt.zero_grad()
        torch.nn.functional.smooth_l1_loss(reference(x), y).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
        refopt.step()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, atol=1e-12, rtol=0)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            accumulated_step(None, None, [], None)
