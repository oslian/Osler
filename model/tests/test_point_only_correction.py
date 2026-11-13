import unittest
import numpy as np
import pandas as pd
from scripts.diagnose_point_only_correction import point_only


class PointTests(unittest.TestCase):
    def test_query_truth_not_used(self):
        rng=np.random.default_rng(19);x=rng.normal(size=200)
        source=pd.DataFrame(dict(real=x,truth=1+2*x+rng.normal(size=200)*.2,fit_scale=1.,component=np.repeat(np.arange(4),50)))
        query=source.copy();a,info=point_only(source,query)
        self.assertEqual(info['status'],'fitted')
        query['truth']=1e9;b,_=point_only(source,query)
        np.testing.assert_array_equal(a,b)
        self.assertLess(np.mean(abs(a-source.truth)),np.mean(abs(source.real-source.truth)))

    def test_unsupported_is_not_success(self):
        source=pd.DataFrame(dict(real=[0.],truth=[1.],fit_scale=[1.],component=[0]))
        p,info=point_only(source,source)
        self.assertIsNone(p)
        self.assertEqual(info['status'],'insufficient_support')


if __name__=='__main__':unittest.main()
