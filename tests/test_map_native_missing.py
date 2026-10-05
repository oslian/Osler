import unittest
import numpy as np
from scripts.train_map_native_missing import missing_view


class MissingTests(unittest.TestCase):
    def test_support_removes_only_other_numbers(self):
        x = np.array([[4.,3.,0.,9.,1.,1.,0.,1.,0.,0.,7.,0.]],dtype=np.float32)
        full = missing_view(x,2,0,np.array([True,True]),'full').reshape(1,3,2,2)
        support = missing_view(x,2,0,np.array([True,True]),'support').reshape(1,3,2,2)
        np.testing.assert_array_equal(full[:,:,:,0],support[:,:,:,0])
        np.testing.assert_array_equal(full[:,1:],support[:,1:])
        np.testing.assert_array_equal(support[:,0,:,1],0.)

    def test_only_masked_numbers_change_and_zero_is_observed(self):
        x = np.array([[0.,0.,0.,1.,7.,0.]],dtype=np.float32)
        original = x.copy()
        v = missing_view(x,2,1,np.array([True,True]),'full')
        self.assertTrue(np.isnan(v[0,0]))
        self.assertEqual(v[0,1],0.)
        np.testing.assert_array_equal(v[:,2:],original[:,2:])
        np.testing.assert_array_equal(x,original)
        np.testing.assert_array_equal(missing_view(x,2,1,np.array([True,True]),'own'),[[0.,1.,0.]])


if __name__ == '__main__':
    unittest.main()
