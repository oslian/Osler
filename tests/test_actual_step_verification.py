import unittest
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from scripts import train_actual_step as audit
from scripts.verify_actual_step import scalar_results,verify_trace,compare


class ActualStepVerificationTest(unittest.TestCase):
    def test_scalar_statistics(self):
        rows=[]
        for fold in range(2):
            for i in range(36):
                rows.append(dict(patient=str(i),component=str(i%3),fold=fold,context=0,variable='map',
                    truth=float(i%4),actual_step=.1*i,**{c:.2*i+j*.01 for j,c in enumerate(audit.CONTROLS)}))
        f=pd.DataFrame(rows);inventory=['map','bun']
        expected=audit.describe(f,inventory);expected['map']=audit.assess(f,audit.base.SEED)
        self.assertLess(compare(expected,scalar_results(f,inventory,True)),1e-12)
        with self.assertRaises(ValueError):scalar_results(pd.concat([f,f.iloc[:1]]),inventory)

    def test_trace_and_tampering(self):
        p=torch.tensor([1.,2.,3.,4.]);q=p-.1;t=torch.zeros(4)
        f=pd.DataFrame(dict(fold=0,epoch=1,step=[0,0,1,1],slot=[0,1,0,1],target=[0,1,0,0],
            cross_target=[True,True,False,False],projected=[True,True,False,False],rounding_veto=False,
            raw_dot=[.1,.1,-.1,-.1],raw_tolerance=1e-12,actual_dot=-.1,actual_tolerance=1e-12,
            raw_norm=1.,actual_norm=[.5,.5,1.,1.],correction_norm=[.5,.5,0.,0.],
            truth=t.numpy(),prediction_before=p.numpy(),prediction_after=q.numpy(),
            loss_before=F.smooth_l1_loss(p,t,reduction='none').numpy(),loss_after=F.smooth_l1_loss(q,t,reduction='none').numpy()))
        # Saved parquet rows hold scalar float64 representations of float32 loss.
        f[['prediction_before','prediction_after','truth','loss_before','loss_after']]=f[
            ['prediction_before','prediction_after','truth','loss_before','loss_after']].astype(float)
        self.assertLess(compare(audit.trace_summary(f),verify_trace(f)),1e-12)
        bad=f.copy();bad.loc[0,'actual_dot']=.1
        with self.assertRaises(ValueError):verify_trace(bad)
        bad=f.copy();bad.loc[0,'loss_after']+=.01
        with self.assertRaises(AssertionError):verify_trace(bad)
        bad=f.copy();bad.loc[0,'raw_norm']=np.nan
        with self.assertRaises(ValueError):verify_trace(bad)


if __name__=='__main__':unittest.main()
