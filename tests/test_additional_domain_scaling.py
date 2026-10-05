import unittest
import numpy as np
import pandas as pd
from scripts.scale_additional_domains import anchor_labels
from scripts.encoder_fold_scaling import fit_scaler
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples


class AdditionalScalingTests(unittest.TestCase):
    def events(self,rows):
        f=pd.DataFrame(rows,columns=['source_id','measured','available','value','quality_valid'])
        f['stay_id']='s';f['variable']='map';f['source_kind']='bedside_available'
        origin=pd.Timestamp('2100-01-01')
        f['measured_at']=origin+pd.to_timedelta(f.measured,unit='h')
        f['event_time']=origin+pd.to_timedelta(f.available,unit='h')
        return f

    def query(self,times):
        return pd.DataFrame(dict(stay_id=['s']*len(times),variable=['map']*len(times),
            t=pd.Timestamp('2100-01-01')+pd.to_timedelta(times,unit='h')))

    def test_future_conflict_not_visible_earlier(self):
        f=self.events([('a',0,0,80,True),('b',0,1,90,True)])
        result=anchor_labels(f,self.query([0,1]))
        self.assertEqual(result[0],80);self.assertTrue(np.isnan(result[1]))

    def test_conflict_before_latest_selection(self):
        f=self.events([('a',.5,1,80,True),('b',.5,.75,90,True),('c',0,1,70,True)])
        self.assertEqual(anchor_labels(f,self.query([1]))[0],70)

    def test_invalid_latest_does_not_restore_old_draw(self):
        f=self.events([('a',0,1,80,True),('b',.5,1,90,False)])
        self.assertTrue(np.isnan(anchor_labels(f,self.query([1]))[0]))

    def test_revisions_fail_closed(self):
        f=self.events([('a',0,0,80,True),('a',0,1,90,True)])
        with self.assertRaisesRegex(ValueError,'Multiple source versions'):
            anchor_labels(f,self.query([1]))

    def test_heldout_values_not_used(self):
        x=np.array([[0.,np.nan],[2.,np.nan],[999.,8.]])
        a=fit_scaler(x,np.array([True,True,False]));x[2]=[-999.,-8.]
        b=fit_scaler(x,np.array([True,True,False]))
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(a['location'][0],1.);self.assertEqual(a['counts'][1],0)

    def test_randomized_exact_label_reference_parity(self):
        variables=('map','bun','creatinine','hemoglobin','hematocrit')
        rng=np.random.default_rng(20260922)
        compared=0
        for trial in range(12):
            rows=[]
            for i in range(50):
                measured=float(rng.integers(0,55))
                rows.append((str(i),measured,measured+float(rng.choice([0,.5,1,2])),
                             float(rng.integers(0,100)),bool(rng.integers(0,5))))
            events=self.events(rows)
            events['variable']=rng.choice(variables,len(events))
            events['source_kind']=rng.choice(['lab_available','bedside_available'],len(events))
            queries=pd.DataFrame([(s,t,v) for s,t in events[['stay_id','event_time']]
                .drop_duplicates().itertuples(index=False,name=None) for v in variables],
                columns=['stay_id','t','variable'])
            fast=anchor_labels(events,queries)
            reference=reconstruction_examples(events,queries,variables,np.zeros(5),
                np.ones(5),np.arange(5),5)
            for example in reference:
                i=example['query_position'];v=variables[example['target_index']]
                if queries.iloc[i].variable!=v:continue
                self.assertEqual(fast[i],example['truth_normalized'])
                compared+=1
        self.assertGreater(compared,100)
