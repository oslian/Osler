import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.evaluate_potassium_tree_transfer import compare


class TreeTransferTest(unittest.TestCase):
    def test_patient_equal_scoring_and_all_controls(self):
        frame=pd.DataFrame({'subject_id':['a','a','a','b'],'truth':[0.,0.,0.,0.]})
        points={key:np.array([1.,1.,1.,3.]) for key in ('full','own','support','persistence')}
        with patch('scripts.evaluate_potassium_tree_transfer.paired',return_value={'checked':True}) as paired:
            result=compare(frame,points,'test')
        self.assertEqual(result['patients'],2)
        self.assertEqual(result['mae']['full'],2.)
        self.assertEqual(paired.call_count,4)
        self.assertEqual(set(result['comparisons']),{'full_vs_own','full_vs_support',
                         'full_vs_persistence','own_vs_persistence'})


if __name__=='__main__':unittest.main()
