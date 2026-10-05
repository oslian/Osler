import unittest
import torch
from whole_body_jepa.native_carry_audit import post_target_direct_carry


class CarryAuditTest(unittest.TestCase):
    def test_post_last_target_only_and_padding(self):
        gates=torch.full((3,5,2),.5)
        active=torch.tensor([[1,1,1,0,1],[1,1,1,1,1],[0,1,0,0,0]],dtype=torch.bool)
        target=torch.tensor([[1,0,0,0,0],[1,0,0,1,0],[0,0,0,0,0]],dtype=torch.bool)
        factors,count,valid=post_target_direct_carry(gates,active,target)
        torch.testing.assert_close(factors,torch.tensor([[.125,.125],[.5,.5],[1.,1.]]))
        self.assertEqual(count.tolist(),[3,1,0]);self.assertEqual(valid.tolist(),[True,True,False])

    def test_rejects_invalid_gate(self):
        with self.assertRaises(ValueError):
            post_target_direct_carry(torch.full((1,1,1),1.1),torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool))
