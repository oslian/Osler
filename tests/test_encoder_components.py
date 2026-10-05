import unittest
import numpy as np
from scripts.audit_encoder_groups import neutralize_component


class ComponentTest(unittest.TestCase):
    def test_only_requested_component_changes(self):
        a={'current':np.array([[2.,3.],[4.,0.]]),
           'mask':np.array([[1.,1.],[1.,0.]]),
           'ages':np.array([[.1,.2],[.3,7.]]),
           'presence':np.ones((2,1)), 'history':np.array([[0,0],[0,1]])}
        a['input']=np.concatenate([a['current'],a['mask'],a['ages'],a['presence']],axis=1)
        selection=np.array([False,True])
        for mode,field in (('value','current'),('age','ages')):
            b=neutralize_component(a,selection,mode)
            for key in ('current','mask','ages','presence','history'):
                if key!=field:np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_array_equal(a[field][:,0],b[field][:,0])
            start=0 if mode=='value' else 4
            np.testing.assert_array_equal(b['input'][:,start:start+2],b[field])
            self.assertEqual(b['current'][1,1],0.)
            self.assertEqual(b['ages'][1,1],7.)
        self.assertEqual(a['current'][0,1],3.)
        self.assertEqual(a['ages'][0,1],.2)


if __name__=='__main__':unittest.main()
