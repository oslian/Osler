import unittest
import numpy as np
from whole_body_jepa.availability_inputs import fit_numeric_rescale,apply_numeric_rescale


class NumericRescaleTest(unittest.TestCase):
    def test_fit_scope_missing_and_target(self):
        a={'current':np.array([[1.,10.],[2.,14.],[3.,999.]],np.float32),
           'mask':np.array([[1.,1.],[1.,1.],[1.,0.]],np.float32),
           'history':np.array([[0,0],[0,1],[1,2]]),'ages':np.ones((3,2))}
        a['input']=np.concatenate([a['current'],a['mask'],a['ages']],axis=1)
        c,s=fit_numeric_rescale(a,np.arange(3),np.array([True,True,False]),0)
        np.testing.assert_allclose(c,[0,12]);np.testing.assert_allclose(s,[1,2])
        b=apply_numeric_rescale(a,c,s)
        np.testing.assert_allclose(b['current'],[[1,-1],[2,1],[3,0]])
        for key in ('mask','history','ages'):np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(b['input'][:,:2],b['current'])


if __name__=='__main__':unittest.main()
