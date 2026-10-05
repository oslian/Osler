import unittest
import numpy as np
from scripts.audit_potassium_raw_history import history_features


class RawHistoryFeaturesTest(unittest.TestCase):
    def test_support_keeps_own_and_masks(self):
        a={'current':np.arange(12).reshape(4,3).astype(float),'mask':np.ones((4,3)),
           'ages':np.full((4,3),.2),'history':np.array([[0,0],[0,1],[1,2],[2,3]])}
        rows=np.array([2,3]);full=history_features(a,rows,1,'full')
        support=history_features(a,rows,1,'support');own=history_features(a,rows,1,'own')
        np.testing.assert_array_equal(support[:,6:],full[:,6:])
        np.testing.assert_array_equal(support[:,[1,4]],own[:,:2])
        self.assertTrue((support[:,[0,2,3,5]]==0).all())
        self.assertEqual(own.shape,(2,6))
        np.testing.assert_array_equal(a['current'],np.arange(12).reshape(4,3))


if __name__=='__main__':unittest.main()
