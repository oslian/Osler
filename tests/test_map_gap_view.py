import unittest
import numpy as np
from scripts.train_map_derived_gap import gap_view


class GapViewTests(unittest.TestCase):
    def test_replacement_is_isolated_and_support_keeps_masks(self):
        pack=dict(variables=['map','anion_gap'],target_index=0,allowed=np.ones(2,bool))
        x=np.array([[3.,6.,1.,1.,0.,0.]],dtype=np.float32)
        d=np.array([[[10.],[1.],[.25]]],dtype=np.float32)
        real=gap_view(x,d,pack,'derived').reshape(1,3,1,2)
        support=gap_view(x,d,pack,'derived_support').reshape(1,3,1,2)
        np.testing.assert_array_equal(real[:,:,:,0],x.reshape(1,3,1,2)[:,:,:,0])
        np.testing.assert_array_equal(real[:,:,:,1],d)
        self.assertEqual(support[0,0,0,1],0.)
        np.testing.assert_array_equal(real[:,1:],support[:,1:])
        self.assertEqual(x[0,1],6.)


if __name__=='__main__':
    unittest.main()
