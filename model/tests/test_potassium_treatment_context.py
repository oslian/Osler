import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.potassium_treatment_context import context_features, shuffled_context, order_record


class TreatmentContextTests(unittest.TestCase):
    def test_simultaneous_rates_do_not_use_id_as_precedence(self):
        q=pd.DataFrame(dict(stay_id=['a'],offset=[10]))
        events=pd.DataFrame([dict(stay_id='a',source_id=str(i),offset=10,action='insulin',dimension='drug_units_per_hour',value=v) for i,v in enumerate([0.,2.])])
        x,c=context_features(q,events);idx=c.index(('insulin','drug_units_per_hour'))
        self.assertTrue(np.isnan(x[0,idx]))
        events['source_id']=['9','1']
        self.assertTrue(np.isnan(context_features(q,events)[0][0,idx]))
        events['value']=0.
        self.assertEqual(context_features(q,events)[0][0,idx],0.)

    def test_order_does_not_use_final_status_or_dose(self):
        row = dict(patientunitstayid=1, medicationid=2, drugorderoffset=10,
                   drugordercancelled='No', dosage='10', drugstopoffset=20)
        first = order_record(row, 'potassium_repletion')
        row.update(drugordercancelled='Yes', dosage='999', drugstopoffset=9999)
        second = order_record(row, 'potassium_repletion')
        for key in ('stay_id','source_id','offset','action','dimension'):
            self.assertEqual(first[key],second[key])
        self.assertTrue(np.isnan(second['value']))
        row['drugorderoffset']=None
        self.assertIsNone(order_record(row,'potassium_repletion'))

    def test_time_zero_and_unit_separation(self):
        q = pd.DataFrame({'stay_id':['a'], 'offset':[400.]})
        e = pd.DataFrame([dict(stay_id='a', source_id=str(i), offset=t, action='potassium_repletion', value=v, dimension='volume_ml_per_hour') for i,t,v in [(1,40,88), (2,399,0), (3,401,999)]])
        x, columns = context_features(q,e)
        self.assertEqual(x[0,columns.index(('potassium_repletion','volume_ml_per_hour'))],0)
        self.assertTrue(np.isnan(x[0,columns.index(('potassium_repletion','charge_meq_per_hour'))]))
        self.assertAlmostEqual(x[0,columns.index(('potassium_repletion','count'))],np.log(2))

    def test_placebo_roles_self_and_future(self):
        q = pd.DataFrame(dict(subject_id=['a','a','b','b','c','d'], offset=[1,3,2,4,1,1]))
        x = np.arange(6,dtype=float)[:,None]
        out = shuffled_context(q,x,[np.array([1,1,1,1,0,0],bool),np.array([0,0,0,0,1,1],bool)],7)
        self.assertTrue(np.isnan(out[0,0]))
        self.assertEqual(out[1,0],2)
        self.assertEqual(out[2,0],0)
        self.assertEqual(out[3,0],1)
        self.assertEqual(out[4,0],5)
        self.assertEqual(out[5,0],4)
