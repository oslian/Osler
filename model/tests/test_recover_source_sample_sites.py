import unittest
import torch
from scripts.recover_source_sample_sites import assert_same_example


class RecoveryTests(unittest.TestCase):
    def test_exact(self):
        a = dict(tokens=torch.tensor([1.,2.]),anchor_hour=2.)
        assert_same_example(a,dict(tokens=a['tokens'].clone(),anchor_hour=2.))

    def test_changed_time_or_value_fails(self):
        a = dict(tokens=torch.tensor([1.,2.]),anchor_hour=2.)
        for b in (dict(tokens=torch.tensor([1.,3.]),anchor_hour=2.),dict(tokens=a['tokens'],anchor_hour=3.)):
            with self.assertRaises(ValueError):
                assert_same_example(a,b)
