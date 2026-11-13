import unittest
from collections import Counter
import numpy as np
import torch
from scripts.train_encoder_crossfit import epoch_plan, fresh_encoder


class TrainingPlanTests(unittest.TestCase):
    def test_fresh_encoder_does_not_load_learned_checkpoint_values(self):
        ck=torch.load('models/whole_body_joint_jepa.pt',map_location='cpu',weights_only=False)
        with torch.random.fork_rng():
            torch.manual_seed(19); first=fresh_encoder(ck)
            structural={'variable_module_ids','variable_module_membership','horizon_bucket_values'}
            for name,value in ck['state_dict'].items():
                if name not in structural:
                    value.fill_(7)
            torch.manual_seed(19); second=fresh_encoder(ck)
        self.assertEqual(first.state_dict().keys(),second.state_dict().keys())
        for name,value in first.state_dict().items():
            torch.testing.assert_close(value,second.state_dict()[name],rtol=0,atol=0)

    def test_same_draws_every_target_and_balanced_available_groups(self):
        pools={'a':{0:[('p',0)],1:[('p',1)]},'b':{0:[('q',0)],1:[]}}
        weights={'a':{0:np.ones(1),1:np.ones(1)},'b':{0:np.ones(1)}}
        p=epoch_plan(pools,weights,np.random.default_rng(19),draws=8)
        self.assertEqual(Counter(c for g,(person,c) in p),{0:8,1:8})
        self.assertEqual(Counter(g for g,(person,c) in p if c==0),{'a':4,'b':4})
        self.assertEqual(p,epoch_plan(pools,weights,np.random.default_rng(19),draws=8))
