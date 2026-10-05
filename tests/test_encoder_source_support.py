import unittest
import numpy as np
from whole_body_jepa.availability_inputs import source_patient_support,restrict_input_support


class SourceSupportTest(unittest.TestCase):
    def test_distinct_fit_patients_not_rows_or_validation(self):
        a={'mask':np.array([[1,0],[1,0],[0,1],[0,1]]),
           'history':np.arange(4).reshape(-1,1)}
        actual=source_patient_support(a,np.arange(4),['a','a','b','c'],np.array([1,1,1,0],bool))
        np.testing.assert_array_equal(actual,[1,1])

    def test_masking_is_consistent_and_does_not_mutate_original(self):
        a={'current':np.ones((2,2)),'mask':np.ones((2,2)),
           'ages':np.zeros((2,2)),'presence':np.ones((2,2)),
           'input':np.ones((2,10)),'history':np.array([[0],[1]])}
        b=restrict_input_support(a,[True,False],np.array([0,1]))
        np.testing.assert_array_equal(b['current'][:,1],0)
        np.testing.assert_array_equal(b['mask'][:,1],0)
        np.testing.assert_array_equal(b['ages'][:,1],7)
        np.testing.assert_array_equal(b['presence'][:,1],0)
        np.testing.assert_array_equal(b['input'][:,2:4],b['mask'])
        np.testing.assert_array_equal(b['input'][:,4:6],b['ages'])
        np.testing.assert_array_equal(b['input'][:,7:9],b['presence'])
        np.testing.assert_array_equal(a['current'],1)


if __name__=='__main__':unittest.main()
