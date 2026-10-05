import unittest
import pandas as pd
from scripts.certify_continuation_selection0 import certify_queries


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.source=pd.DataFrame(dict(patient=['a','b'],target=['x','x'],fold=0,
            truth=[1.,2.],component=['1','1'],fit_scale=1.,population=1.,real=[1.,2.],process=1.))
        self.base=self.source.assign(fold=2)
        self.fresh=self.source[['patient','target','fold']].assign(fold=2,real10=[1.1,2.1],process10=1.)
        self.sroles=dict(a='selection',b='selection',c='fit')
        self.droles=dict(a='test',b='test',c='fit')

    def test_exact_reuse_keeps_values(self):
        result=certify_queries(self.source,self.base,self.fresh,self.sroles,self.droles)
        self.assertTrue(result.fold.eq(0).all())
        self.assertEqual(result.real10.tolist(),self.fresh.real10.tolist())
        self.assertTrue(self.fresh.fold.eq(2).all())

    def test_wrong_fit_or_query_role_rejected(self):
        for person,role in [('a','fit'),('c','selection')]:
            roles=dict(self.droles);roles[person]=role
            with self.assertRaises(ValueError):certify_queries(self.source,self.base,self.fresh,self.sroles,roles)

    def test_changed_values_or_order_rejected(self):
        base=self.base.copy();base.loc[0,'truth']=7
        with self.assertRaises(AssertionError):certify_queries(self.source,base,self.fresh,self.sroles,self.droles)
        with self.assertRaises(AssertionError):certify_queries(self.source,self.base,self.fresh.iloc[::-1],self.sroles,self.droles)

    def test_nonfinite_prediction_rejected(self):
        fresh=self.fresh.copy();fresh.loc[0,'real10']=float('nan')
        with self.assertRaises(ValueError):certify_queries(self.source,self.base,fresh,self.sroles,self.droles)


if __name__=='__main__':unittest.main()
