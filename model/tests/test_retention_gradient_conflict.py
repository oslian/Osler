import unittest
import torch
from scripts.audit_retention_gradient_conflict import geometry


class GeometryTest(unittest.TestCase):
    def test_opposed(self):
        result = geometry([torch.tensor([1., 0.])], [torch.tensor([-2., 0.])])
        self.assertEqual(result['cosine'], -1.)
        self.assertEqual(result['retention_to_primary'], 2.)

    def test_orthogonal(self):
        self.assertEqual(geometry([torch.tensor([1., 0.])], [torch.tensor([0., 1.])])['cosine'], 0.)

    def test_unused(self):
        result = geometry([torch.tensor([1.]), None], [None, torch.tensor([2.])])
        self.assertEqual(result['cosine'], 0.)
        self.assertIsNone(geometry([None], [None])['cosine'])
