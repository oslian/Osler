import ast
import inspect
import unittest
import torch
from scripts import train_reconstruction_replication_v2 as runner


class ProcessWiringTest(unittest.TestCase):
    def test_views(self):
        x=torch.arange(20,dtype=torch.float32).reshape(2,10)
        e={'tokens':x}
        for mode in ('real','placebo'):
            self.assertTrue(torch.equal(runner.input_tokens(e,mode,2),x))
        p=runner.input_tokens(e,'process',2)
        self.assertTrue((p[:,:2]==0).all())
        self.assertTrue(torch.equal(p[:,2:],x[:,2:]))

    def test_fit_and_tune_both_use_same_view(self):
        tree=ast.parse(inspect.getsource(runner.main))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model']
        self.assertEqual(len(calls),2)
        for call in calls:
            self.assertIsInstance(call.args[0],ast.Call)
            self.assertEqual(call.args[0].func.id,'input_tokens')
