import unittest
import numpy as np
from scripts.audit_auxiliary_learning import own_features


class OwnFeatureTests(unittest.TestCase):
    def test_no_cross_variable_or_future_access(self):
        a={key:np.arange(12,dtype=float).reshape(4,3) for key in ('current','mask','ages')}
        a['history']=np.array([[0,0],[0,1],[1,2],[2,3]])
        rows=np.array([2])
        before=own_features(a,rows,1,True)
        for key in ('current','mask','ages'):
            a[key][:,[0,2]]=999
            a[key][3,1]=999
        np.testing.assert_array_equal(before,own_features(a,rows,1,True))
        np.testing.assert_array_equal(before,np.array([[4,7,4,7,4,7]]))

    def test_current_does_not_use_history(self):
        a={key:np.arange(6).reshape(3,2) for key in ('current','mask','ages')}
        np.testing.assert_array_equal(own_features(a,np.array([1]),0,False),[[2,2,2]])
