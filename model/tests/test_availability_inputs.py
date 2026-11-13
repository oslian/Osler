import unittest
import pandas as pd
import numpy as np
import torch
from types import SimpleNamespace
from whole_body_jepa.availability_inputs import snapshots,filter_units,build_arrays


class AvailabilityInputsTest(unittest.TestCase):
    def raw(self):
        t=pd.Timestamp('2100-01-01')
        return pd.DataFrame([dict(stay_id='s',source_id=i,variable=v,value=x,unit=u,
            measured_at=t+pd.Timedelta(hours=m),available_at=t+pd.Timedelta(hours=a))
            for i,v,x,u,m,a in [('1','glucose',100,'mg/dL',0,2),('2','potassium',4,'mmol/L',1,3),
                                ('3','glucose',80,'mg/dL',-1,4)]])

    def test_each_historical_time_has_own_availability(self):
        times=pd.date_range('2100-01-01',periods=5,freq='h')
        values,ages=snapshots(self.raw(),times,['glucose','potassium'],6)
        self.assertTrue(np.isnan(values[:2]).all())
        self.assertTrue(np.isnan(values[2,1]))
        self.assertEqual(values[4,0],100)
        self.assertEqual(ages[4,0],4)
        self.assertEqual(values[3,1],4)

    def test_unknown_unit_excluded(self):
        raw=self.raw()
        raw.loc[0,'unit']='unknown'
        got=filter_units(raw,{'glucose':(10,2000),'potassium':(1,10)})
        self.assertNotIn('1',set(got.source_id))

    def test_conflict_known_time_and_resolution(self):
        t=pd.Timestamp('2100-01-01')
        raw=pd.DataFrame([('a',0,0,2.),('b',0,1,5.),('b',0,2,2.),('c',3,3,0.)],
            columns=['source_id','measured_at','available_at','value'])
        raw['stay_id']='s';raw['variable']='x'
        for key in ('measured_at','available_at'):
            raw[key]=t+pd.to_timedelta(raw[key],unit='h')
        times=pd.date_range(t,periods=4,freq='h')
        values,ages=snapshots(raw,times,['x'],6)
        np.testing.assert_equal(values[:,0],[2.,np.nan,2.,0.])
        np.testing.assert_equal(ages[:,0],[0.,1.,2.,0.])
        x,_=snapshots(raw.sample(frac=1,random_state=7),times,['x'],6)
        np.testing.assert_equal(x,values)
        prefix,_=snapshots(raw.loc[raw.available_at<=times[1]],times[:2],['x'],6)
        np.testing.assert_equal(prefix,values[:2])

    def test_future_events_do_not_change_prefix(self):
        raw=self.raw()
        times=pd.date_range('2100-01-01',periods=3,freq='h')
        x,_=snapshots(raw,times,['glucose','potassium'],6)
        y,_=snapshots(raw.loc[raw.available_at<=times[-1]],times,['glucose','potassium'],6)
        np.testing.assert_equal(x,y)

    def test_micro_unit_alias(self):
        raw=self.raw().iloc[:1].copy()
        raw['variable']='wbc'
        raw['unit']='K/mcL'
        raw['value']=7.
        self.assertEqual(len(filter_units(raw,{'wbc':(.1,300)})),1)

    def test_encoder_history_masks_and_width(self):
        class Scaler:
            def normalize_current(self,f):
                x=f[['glucose_t','potassium_t']].to_numpy(dtype=np.float32)
                a=f[['glucose_age_hr','potassium_age_hr']].to_numpy(dtype=np.float32)
                return np.nan_to_num(x),np.isfinite(x).astype(np.float32),np.nan_to_num(a,nan=168)/24
        assets=SimpleNamespace(checkpoint={'variables':['glucose','potassium'],'modules':['a','b'],'input_dim':10},
            scaler=Scaler(),model=SimpleNamespace(variable_module_ids=torch.tensor([0,1])),max_age_hours=6)
        origin=pd.Timestamp('2100-01-01')
        frame=pd.DataFrame({'stay_id':['s','s'],'t':[origin+pd.Timedelta(hours=h) for h in (2,3)]})
        arrays,rows=build_arrays(frame,self.raw(),assets,{'s':origin})
        self.assertEqual(arrays['input'].shape[1],10)
        np.testing.assert_equal(arrays['mask'][rows],[[1,0],[1,1]])
        self.assertTrue((arrays['mask'][arrays['history'][rows,0]]==0).all())
        self.assertEqual(arrays['current'][rows[0],1],0)
        self.assertEqual(arrays['ages'][rows[0],0],2/24)
        timed,timed_rows=build_arrays(frame,self.raw(),assets,{'s':origin},query_horizons=[5.25,6.75])
        np.testing.assert_array_equal(rows,timed_rows)
        np.testing.assert_allclose(timed['horizon'][rows],[5.25,6.75])
        for key in arrays:
            if key!='horizon':np.testing.assert_array_equal(arrays[key],timed[key])
        for invalid in ([0,6],[np.nan,6],[6],[-1,6]):
            with self.assertRaises(ValueError):
                build_arrays(frame,self.raw(),assets,{'s':origin},query_horizons=invalid)


if __name__=='__main__': unittest.main()
