from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from scripts import fit_tangent_bridge as audit
from scripts.verify_tangent_bridge import scalar_results,check_line
from scripts.verify_frozen_bridge import compare


class TangentVerificationTest(unittest.TestCase):
    def frame(self,n=33):
        return pd.DataFrame([dict(patient=f'p{i:03d}',component=f'c{i%3}',fold=0,context=0,
            variable='map',truth=1.,tangent_step=.5,local_linear=.6,
            **{c:0. for c in audit.CONTROLS}) for i in range(n)])

    def test_full_scalar_statistics_and_support(self):
        f=self.frame();inventory=['map','bun']
        expected=audit.describe(f,inventory)
        expected['map']=audit.assess(f,audit.base.SEED)
        self.assertLess(compare(scalar_results(f,inventory,True),expected),1e-12)
        self.assertTrue(all(c['screen_pass'] for c in expected['map']['comparisons'].values()))
        f.component='one'
        self.assertFalse(any(c['screen_pass'] for c in scalar_results(f,inventory,True)['map']['comparisons'].values()))

    def test_invalid_saved_step_rejected(self):
        f=self.frame(3);f['tangent_base']=0.;f['tangent_slope']=0.;f['tangent_step']=0.;f['local_linear']=0.
        info=dict(accepted_alpha=0.,linear_fit_loss=.5,solver=dict(converged=True,gradient_inf=0.,
            regularization=1.,final_objective=.5),line_search=dict(origin=.5,derivative=0.,trials=[]))
        check_line(info,f,np.zeros(2))
        altered=deepcopy(info);altered['accepted_alpha']=1.
        with self.assertRaises(ValueError):check_line(altered,f,np.zeros(2))
        f['local_linear']=.2
        with self.assertRaises(AssertionError):check_line(info,f,np.zeros(2))


if __name__=='__main__':unittest.main()
