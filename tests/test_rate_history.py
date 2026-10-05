import unittest
import pandas as pd
import numpy as np
from whole_body_jepa.potassium_treatment_context import rate_history_features, rate_transition_features, context_features, changed_rate_brackets


class RateHistoryTests(unittest.TestCase):
    def test_changed_pair_brackets(self):
        self.assertEqual(changed_rate_brackets([10,30],[1,2],20),(False,True))
        self.assertEqual(changed_rate_brackets([20,30],[1,2],20),(True,False))
        self.assertEqual(changed_rate_brackets([10,20],[1,2],20),(False,False))
        self.assertEqual(changed_rate_brackets([10,30],[0,0],20),(False,False))
        with self.assertRaises(ValueError):changed_rate_brackets([10,10],[1,2],20)

    def test_internal_transition_collision_and_repair(self):
        q=pd.DataFrame(dict(stay_id=['a'],offset=[360]))
        def series(values):
            return pd.DataFrame([dict(stay_id='a',source_id=str(i),offset=t,action='insulin',dimension='drug_units_per_hour',value=v,original_label='insulin') for i,(t,v) in enumerate(zip([60,120,180,300],values))])
        a,b=series([0,2,0,2]),series([0,0,2,2])
        for fn in (context_features,rate_history_features):
            np.testing.assert_allclose(fn(q,a)[0],fn(q,b)[0],equal_nan=True)
        x,c=rate_transition_features(q,a);y,_=rate_transition_features(q,b)
        def get(z,k):return z[0,c.index(('insulin','drug_units_per_hour',k))]
        self.assertEqual(get(x,'change_count'),3)
        self.assertEqual(get(y,'change_count'),1)
        self.assertEqual(get(x,'reversal_count'),2)
        self.assertEqual(get(x,'last_change_age_hours'),1)
        self.assertEqual(get(y,'last_change_age_hours'),3)
        duplicate=pd.concat([a,a.iloc[[1]],series([0,2,0,999]).iloc[[-1]].assign(offset=361)])
        np.testing.assert_allclose(x,rate_transition_features(q,duplicate)[0],equal_nan=True)

    def test_zero_future_and_ambiguous_series(self):
        q=pd.DataFrame(dict(stay_id=['a'],offset=[400]))
        e=pd.DataFrame([dict(stay_id='a',offset=t,action='insulin',dimension='drug_units_per_hour',value=v,original_label='insulin') for t,v in [(100,4),(390,0),(401,999)]])
        x,c=rate_history_features(q,e)
        j=c.index(('insulin','drug_units_per_hour','change'))
        self.assertEqual(x[0,j],-4)
        e.loc[0,'original_label']='another label'
        self.assertTrue(np.isnan(rate_history_features(q,e)[0][0,j]))
