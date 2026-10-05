import unittest
import numpy as np
import pandas as pd
from scripts.train_map_native_event_weights import event_weights


class EventWeightTests(unittest.TestCase):
    def test_patient_event_hierarchy_and_stay_namespace(self):
        f = pd.DataFrame({'subject_id':['a','a','a','b'],
                          'stay_id':['x','x','y','z'], 'future_id':['1']*4})
        np.testing.assert_allclose(event_weights(f),[.5,.5,1.,2.])
        np.testing.assert_allclose(event_weights(f.iloc[::-1]),[2.,1.,.5,.5])

    def test_missing_identity_rejected(self):
        f = pd.DataFrame({'subject_id':['a'],'stay_id':['s'],'future_id':['']})
        with self.assertRaises(ValueError):
            event_weights(f)


if __name__ == '__main__':
    unittest.main()
