import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples
from whole_body_jepa.reconstruction_support import reconstruction_support


class ReconstructionSupportTests(unittest.TestCase):
    variables=('bun','creatinine','map','hemoglobin','hematocrit')

    def compare(self, rows):
        origin=pd.Timestamp('2100-01-01')
        events=pd.DataFrame(rows,columns=['source_id','variable','measured','available','value','quality_valid'])
        events['stay_id']='s'; events['source_kind']='lab_available'
        events['measured_at']=origin+pd.to_timedelta(events.measured,unit='h')
        events['event_time']=origin+pd.to_timedelta(events.available,unit='h')
        events=events.drop(columns=['measured','available'])
        queries=pd.DataFrame([(s,t,v) for s,t in events[['stay_id','event_time']].drop_duplicates().itertuples(index=False,name=None)
                              for v in self.variables],columns=['stay_id','t','variable'])
        actual=reconstruction_support(events,queries,self.variables)
        reference=list(reconstruction_examples(events,queries,self.variables,np.zeros(5),np.ones(5),np.arange(5),5))
        expected={(e['query_position'], self.variables[e['target_index']]):e for e in reference}
        for i,q in enumerate(queries.itertuples()):
            example=expected.get((i,q.variable))
            self.assertEqual(actual.iloc[i].status=='accepted',example is not None)
            if example is not None:
                mask=example['tokens'][:,5:10].bool()
                self.assertEqual(actual.iloc[i].independent_variables,int(mask.any(0).sum()))
                self.assertEqual(actual.iloc[i].independent_event_groups,len(example['tokens']))
        return actual

    def test_revision_conflict_quality_and_siblings(self):
        self.compare([('a','map',0,0,80,True),('b','bun',0,0,20,True),
                      ('c','creatinine',1,1,1,True),('a','map',0,2,90,True),
                      ('d','map',0,2,80,True),('e','hemoglobin',2,2,10,True),
                      ('f','hematocrit',2,2,30,False),('g','bun',2,4,25,True)])

    def test_old_conflict_future_and_window_boundary(self):
        self.compare([('a','map',0,0,80,True),('b','map',0,47,90,True),
                      ('c','bun',48,48,20,True),('d','hematocrit',48,48,30,True),
                      ('e','hemoglobin',49,49,10,True),('f','map',49,50,70,True)])

    def test_randomized_reference_parity(self):
        rng=np.random.default_rng(7)
        for trial in range(8):
            rows=[]
            for i in range(20):
                t=float(rng.integers(0,60))
                rows.append((str(i),rng.choice(self.variables),t,t+float(rng.choice([0,.5,2])),
                             float(rng.integers(0,100)),bool(rng.integers(0,5))))
            self.compare(rows)
