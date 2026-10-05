import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.change_evaluation import paired_model_errors


class CommonBootstrapTest(unittest.TestCase):
    def test_name_and_row_order_invariant(self):
        frame=pd.DataFrame(dict(subject_id=np.repeat(np.arange(40),2),truth=np.arange(80,dtype=float)))
        a=frame.truth.to_numpy()+np.repeat(np.linspace(-2,3,40),2)
        b=frame.truth.to_numpy()+1
        first=paired_model_errors(frame,dict(candidate=a,baseline=b),'baseline',seed=19,repetitions=1000)
        renamed=paired_model_errors(frame,dict(new_name=a,other=b),'other',seed=19,repetitions=1000)
        self.assertEqual(first['candidate'],renamed['new_name'])
        shuffled=paired_model_errors(frame.iloc[::-1].reset_index(drop=True),dict(candidate=a[::-1],baseline=b[::-1]),'baseline',seed=19,repetitions=1000)
        self.assertEqual(first,shuffled)

    def test_missing_predictions_fail_instead_of_changing_cohort(self):
        frame=pd.DataFrame(dict(subject_id=['a','b'],truth=[1.,2.]))
        with self.assertRaises(ValueError):
            paired_model_errors(frame,dict(a=[1.,np.nan],b=[1.,2.]),'b',seed=1)


if __name__=='__main__':unittest.main()
