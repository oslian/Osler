import unittest
from types import SimpleNamespace

import numpy as np

from whole_body_jepa.native_numeric_control import without_other_numbers


class NativeNumericControlTest(unittest.TestCase):
    def test_preserves_activation_timing_target_and_original(self):
        n = 3
        x = np.arange(4*18, dtype=np.float32).reshape(4,18)
        x[0] = 0
        x[:, n:2*n] = [[0,0,0],[0,1,0],[1,0,0],[0,0,0]]
        x[:, -2] = [0,0,0,1]
        q = dict(current=x[:,:n].copy(), input=x[:,:11].copy(),
                 mask=x[:,n:2*n].copy(), ages=x[:,2*n:3*n].copy())
        original = SimpleNamespace(input=x, query=q, sequences=[np.array([1,2,3])])
        result = without_other_numbers(original,1)
        np.testing.assert_array_equal(result.input[:,1],x[:,1])
        np.testing.assert_array_equal(result.input[:,n:-7],x[:,n:-7])
        np.testing.assert_array_equal(result.input[:,-4:],x[:,-4:])
        np.testing.assert_array_equal(result.query['mask'],q['mask'])
        np.testing.assert_array_equal(result.query['ages'],q['ages'])
        for a in (x,result.input):
            active = a[:,n:2*n].astype(bool).any(axis=1) | a[:,-2].astype(bool)
            np.testing.assert_array_equal(active,[False,True,True,True])
        self.assertTrue((result.input[:,[0,2]]==0).all())
        self.assertTrue((result.input[:,-7:-4]==0).all())
        self.assertTrue((result.query['current'][:,[0,2]]==0).all())
        self.assertGreater(float(x[:,0].sum()),0)
        self.assertIs(result.sequences,original.sequences)

    def test_invalid_target(self):
        storage=SimpleNamespace(query={'current':np.zeros((1,3))})
        with self.assertRaises(ValueError):
            without_other_numbers(storage,3)


if __name__ == '__main__':
    unittest.main()
