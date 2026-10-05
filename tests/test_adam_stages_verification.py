import unittest
import pandas as pd
import torch
from torch.nn import functional as F

from scripts.diagnose_adam_stages import STAGES,summarize
from scripts.verify_adam_stages import scalar_summary,finite_loss_parity
from scripts.verify_frozen_bridge import compare


class VerificationTest(unittest.TestCase):
    def frame(self):
        rows=[]
        for i,flags in enumerate(([False]*5,[False,True,False,True,True],[True,False,False,False,False])):
            r=dict(unscaled_proposal_dot=float(flags[0]),unscaled_proposal_tolerance=1e-12,
                   maximum_coordinate_roundoff=1e-10,truth=.25)
            for s,h in zip(STAGES,flags):r.update({s+'_dot':1. if h else -1.,s+'_tolerance':1e-12})
            for suffix,value in (('before',float(i)),('after',float(i)+.1)):
                p=torch.tensor(value);t=torch.tensor(.25)
                r.update({'prediction_'+suffix:float(p),'loss_'+suffix:float(F.smooth_l1_loss(p,t)),
                          'mae_'+suffix:float((p-t).abs())})
            rows.append(r)
        return pd.DataFrame(rows)

    def test_independent_counts_and_finite_formula(self):
        f=self.frame();finite_loss_parity(f)
        self.assertLess(compare(scalar_summary(f),summarize(f)),1e-12)
        r=scalar_summary(f)
        self.assertEqual(r['first_harm_among_actual_harm']['momentum'],1)
        self.assertEqual(r['stage_transitions']['preconditioned']['harm_to_safe'],1)
        self.assertEqual(r['stage_transitions']['with_decay']['safe_to_harm'],1)

    def test_tamper_rejected(self):
        f=self.frame();f.loc[0,'loss_after']+=.01
        with self.assertRaises(AssertionError):finite_loss_parity(f)
        f=self.frame();f.loc[0,'momentum_tolerance']=0
        with self.assertRaises(ValueError):scalar_summary(f)
        self.assertEqual(scalar_summary(f.iloc[:0]),dict(status='unsupported',examples=0))


if __name__=='__main__':unittest.main()
