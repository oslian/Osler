import unittest
import torch
from scripts.train_native_private_update import private_parameters


class PrivateUpdateTest(unittest.TestCase):
    def test_only_target_modules_train(self):
        model=torch.nn.Module()
        model.shared=torch.nn.Linear(2,2)
        for name in ('target_future_predictors','target_temporal_adapters','value_heads'):
            setattr(model,name,torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(2)]))
        params=private_parameters(model,1)
        self.assertEqual(len(params),6)
        self.assertFalse(model.training)
        for name,p in model.named_parameters():
            self.assertEqual(p.requires_grad,'.1.' in name)
