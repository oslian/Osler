import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.derived_anion_events import derive_events,VARIABLE
from whole_body_jepa.availability_inputs import snapshots


class DerivedEventTests(unittest.TestCase):
    def test_availability_revision_conflict_zero_and_new_incomplete_panel(self):
        origin = pd.Timestamp('2100-01-01')
        raw = pd.DataFrame([
            ('na','sodium',0,0,140.),('cl','chloride',0,1,106.),
            ('bc','bicarbonate',0,2,24.),('bc2','bicarbonate',0,3,25.),
            ('bc2','bicarbonate',0,4,24.),('na','sodium',0,5,130.),
            ('na_new','sodium',6,6,139.),
        ],columns=['source_id','variable','measured_at','available_at','value'])
        for col in ('measured_at','available_at'):
            raw[col] = origin+pd.to_timedelta(raw[col],unit='h')
        raw['stay_id']='s'
        raw['unit']='mmol/L'
        before=raw.copy(deep=True)
        events=derive_events(raw)
        values,_=snapshots(events,pd.date_range(origin,periods=7,freq='h'),[VARIABLE],12)
        np.testing.assert_allclose(values[:,0],[np.nan,np.nan,10.,np.nan,10.,0.,np.nan],equal_nan=True)
        pd.testing.assert_frame_equal(raw,before)

    def test_wrong_unit_blocks_and_different_draws_do_not_merge(self):
        raw=pd.DataFrame({'stay_id':['s']*3,'source_id':['a','b','c'],
            'variable':['sodium','chloride','bicarbonate'],'value':[140.,106.,24.],
            'measured_at':pd.to_datetime(['2100-01-01']*3),
            'available_at':pd.to_datetime(['2100-01-01']*3),'unit':['mg/dL','mmol/L','mmol/L']})
        self.assertFalse(derive_events(raw).quality_valid.any())
        raw['unit']='mmol/L'
        raw.loc[2,'measured_at']+=pd.Timedelta(minutes=1)
        raw.loc[2,'available_at']+=pd.Timedelta(minutes=1)
        self.assertFalse(derive_events(raw).quality_valid.any())


if __name__ == '__main__':
    unittest.main()
