import unittest
import numpy as np
import pandas as pd
from scripts.infer_additional_calibration import verify_roles


class CalibrationRoleTests(unittest.TestCase):
    def test_exact_calibration_only(self):
        scaling=dict(people=['a','b'],variables=['x'],centers=np.array([[0.],[2.]]))
        roles={'a':'calibration','b':'test'}
        frame=pd.DataFrame(dict(patient=['a'],target=['x'],truth=[0.],fold=[2],
            component=['g'],fit_scale=[1.],real=[0.],process=[1.],population=[1.]))
        verify_roles(frame,scaling,roles,2,{'a':'g','b':'h'})
        wrong=frame.copy();wrong['patient']='b';wrong['truth']=2
        with self.assertRaises(ValueError):verify_roles(wrong,scaling,roles,2,{'a':'g','b':'h'})
        wrong=frame.copy();wrong['truth']=1
        with self.assertRaises(ValueError):verify_roles(wrong,scaling,roles,2,{'a':'g','b':'h'})
        with self.assertRaises(ValueError):verify_roles(pd.concat([frame,frame]),scaling,roles,2,{'a':'g'})

    def test_component_and_scale(self):
        scaling=dict(people=['a'],variables=['x'],centers=np.array([[0.]]))
        frame=pd.DataFrame(dict(patient=['a'],target=['x'],truth=[0.],fold=[0],
            component=['wrong'],fit_scale=[1.],real=[0.],process=[1.],population=[1.]))
        with self.assertRaises(ValueError):verify_roles(frame,scaling,{'a':'calibration'},0,{'a':'g'})
        frame['component']='g';frame['fit_scale']=0
        with self.assertRaises(ValueError):verify_roles(frame,scaling,{'a':'calibration'},0,{'a':'g'})


if __name__=='__main__':unittest.main()
