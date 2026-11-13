import unittest
import numpy as np
import torch
from scripts.reuse_equivalent_encoder_fold import assert_same


class FoldEquivalenceTests(unittest.TestCase):
    def test_nested_optimizer_and_rng_equality(self):
        a={'a':[torch.tensor([1.,2.]),np.array([4,5])],'rng':{'x':12}}
        b={'a':[torch.tensor([1.,2.]),np.array([4,5])],'rng':{'x':12}}
        assert_same(a,b)
        b['a'][0][0]=3
        with self.assertRaises(ValueError):assert_same(a,b)

    def test_different_order_or_support_rejected(self):
        with self.assertRaises(ValueError):assert_same(['a','b'],['b','a'])
        with self.assertRaises(ValueError):assert_same({'x':1},{'x':1,'y':0})
