import unittest
import pandas as pd
from scripts.verify_continuation_selection import verify_frames


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.base=pd.DataFrame(dict(patient=['a','b'],target=['x','y'],fold=2))
        self.fresh=self.base.assign(real10=1.,process10=2.)
        self.roles=dict(a='selection',b='selection')
        self.expected={('a','x'),('b','y')}
        self.meta=dict(fold=2,source_fold=0,role='selection',epoch=10,candidate_only=True,
                       promotion_allowed=False,examples=2,patients=2,targets=2)

    def test_correct_reused_fit(self):
        verify_frames(self.base,self.fresh,self.meta,self.roles,2,self.expected)

    def test_incomplete_but_matching_grids_rejected(self):
        meta={**self.meta,'examples':1,'patients':1,'targets':1}
        with self.assertRaises(ValueError):
            verify_frames(self.base.iloc[:1],self.fresh.iloc[:1],meta,self.roles,2,self.expected)

    def test_wrong_role_model_or_promotion_rejected(self):
        for k,v in [('role','test'),('source_fold',2),('promotion_allowed',True),('patients',3)]:
            with self.assertRaises(ValueError):verify_frames(self.base,self.fresh,{**self.meta,k:v},self.roles,2,self.expected)

    def test_nonfinite_rejected(self):
        fresh=self.fresh.copy();fresh.loc[0,'real10']=float('inf')
        with self.assertRaises(ValueError):verify_frames(self.base,fresh,self.meta,self.roles,2,self.expected)

    def test_fit_patient_rejected(self):
        with self.assertRaises(ValueError):verify_frames(self.base,self.fresh,self.meta,{**self.roles,'a':'fit'},2,self.expected)


if __name__=='__main__':unittest.main()
