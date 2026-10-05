import unittest
import numpy as np
import pandas as pd

from scripts.diagnose_nonlinear_source_components import patient_losses, diagnose
from tests.test_nonlinear_placebo_scoring import example


class NonlinearComponentTests(unittest.TestCase):
    def test_patient_errors_average_losses_not_predictions(self):
        a=example(); b=example(); b['fold']=1; b.loc[0,'body_probe']=-3
        errors=patient_losses(pd.concat([a,b]))
        self.assertEqual(len(errors),2)
        self.assertEqual(errors.loc['a','body_probe'],2)

    def test_supported_harm_and_unknown_singleton_are_preserved(self):
        ids=[f'p{i}' for i in range(100)]
        f=pd.DataFrame(dict(component=['a']*60+['b']*39+['c'],body_probe=np.ones(100),native=np.r_[np.zeros(60),np.full(40,2.)]),index=ids)
        r=diagnose(f,'native',19,repetitions=50)
        self.assertEqual(r['supported_nominal_harm_groups'],1)
        self.assertEqual(r['supported_both_half_harm_groups'],1)
        self.assertIsNone(r['variance']['estimated_sampling_variance'])
        self.assertEqual(len(r['groups']),3)

    def test_same_pair_duplication_and_outer_role_rejected(self):
        f=example()
        with self.assertRaises(ValueError): patient_losses(pd.concat([f,f]))
        f['role']='test'
        with self.assertRaises(ValueError): patient_losses(f)


if __name__=='__main__': unittest.main()
