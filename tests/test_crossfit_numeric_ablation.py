import unittest
import torch
import pandas as pd
from scripts.audit_crossfit_numeric_ablation import numeric_ablation, scores


class NumericAblationTests(unittest.TestCase):
    def test_only_requested_numbers_change(self):
        x = torch.arange(24.).reshape(3, 8)
        before = x.clone(); altered = numeric_ablation(x, 1)
        self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.equal(altered[:, 2:], x[:, 2:]))
        self.assertTrue(torch.equal(altered[:, 0], x[:, 0]))
        self.assertTrue((altered[:, 1] == 0).all())

    def test_negative_difference_means_improvement(self):
        f = pd.DataFrame(dict(real=[2., 2.], truth=[0., 0.], process=[3., 3.],
            population=[4., 4.], component=['a', 'b'], bicarbonate=[1., 1.],
            bicarbonate_observed=[True, True]))
        r = scores(f, 'bicarbonate')
        self.assertEqual(r['delta_vs_real'], -1.)
        self.assertEqual(r['delta_vs_process'], -2.)
