import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from whole_body_jepa.raw_lab_replay import canonical_events, build_replay, RawLabResidual, predictions
from scripts.audit_raw_lab_identity import extract_events, audit_and_sequences


class RawLabReplayTests(unittest.TestCase):
    def setUp(self):
        self.t=pd.Timestamp('2020-01-01')
        self.events=pd.DataFrame({'stay_id':['a','a'], 'source_id':['one','two'],
            'measured_at':[self.t,self.t+pd.Timedelta(hours=1)],
            'available_at':[self.t+pd.Timedelta(minutes=30),self.t+pd.Timedelta(hours=2)],'value':[100.,100.]})
        self.q=pd.DataFrame({'t':[self.t+pd.Timedelta(hours=h) for h in (1,3)],'frozen':[100.,100.]})

    def test_event_id_not_value_deduplication(self):
        raw=pd.concat([self.events,self.events],ignore_index=True)
        clean=canonical_events(raw)
        self.assertEqual(len(clean),2)
        seq=build_replay(raw,self.q,center=100,scale=30)
        self.assertEqual(seq.updates.sum(),2)

    def test_conflict_rejected_and_revisions_retained(self):
        conflict=self.events.iloc[[0]].assign(value=200.)
        with self.assertRaises(ValueError): canonical_events(pd.concat([self.events,conflict]))
        revision=conflict.assign(available_at=self.t+pd.Timedelta(hours=1))
        self.assertEqual(len(canonical_events(pd.concat([self.events,revision]))),3)

    def test_future_and_unknown_availability(self):
        seq=build_replay(self.events,self.q.iloc[:1],center=100,scale=30)
        self.assertEqual(seq.updates.sum(),1)
        self.assertEqual(len(canonical_events(self.events.assign(available_at=pd.NaT))),0)

    def test_variable_collision_cannot_be_silently_deduplicated(self):
        event=self.events.iloc[:1].assign(variable='glucose')
        other=event.assign(variable='sodium')
        with self.assertRaises(ValueError):
            canonical_events(pd.concat([event,other]))
        with self.assertRaises(ValueError):
            canonical_events(pd.concat([event,other.assign(available_at=self.t+pd.Timedelta(hours=2))]))
        different_stay=other.assign(stay_id='b')
        self.assertEqual(len(canonical_events(pd.concat([event,different_stay]))),2)

    def test_duplicate_query_does_not_change_state(self):
        torch.manual_seed(19)
        model=RawLabResidual()
        torch.nn.init.normal_(model.head[-1].weight,std=.1)
        seq=build_replay(self.events,self.q,center=100,scale=30)
        a=predictions(model,[seq],self.q,30.)
        q=pd.concat([self.q.iloc[:1],self.q],ignore_index=True)
        seq2=build_replay(self.events,q,center=100,scale=30)
        b=predictions(model,[seq2],q,30.)
        np.testing.assert_allclose(a,b[[1,2]],atol=1e-6)
        self.assertEqual(seq2.updates.sum(),2)

    def test_raw_extraction_preserves_available_time_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            raw=pd.DataFrame({'labid':[11,12], 'patientunitstayid':[1,1],
                'labresultoffset':[0,60], 'labresultrevisedoffset':[30,120],
                'labname':['glucose']*2,'labresult':[100.,100.],
                'labmeasurenamesystem':['mg/dL']*2,'labmeasurenameinterface':['mg/dL']*2})
            raw.to_csv(root/'lab.csv.gz',index=False)
            result=extract_events([pd.DataFrame({'stay_id':['eicu:1']})],'eicu',root,root/'cache.parquet')
            self.assertEqual(result.source_id.nunique(),2)
            np.testing.assert_allclose((result.available_at-result.measured_at).dt.total_seconds(),[1800,3600])

    def test_late_current_is_not_legal_even_when_value_matches(self):
        query=pd.DataFrame({'subject_id':['p'], 'stay_id':['a'], 't':[self.t],
            'current_event_time':[self.t], 'current':[100.], 'truth':[100.],
            'future_event_time':[self.t+pd.Timedelta(hours=1)], 'frozen':[100.]})
        frame,_,audit=audit_and_sequences(query,self.events,100.,30.)
        self.assertFalse(frame.strict.any())
        self.assertEqual(audit['scored_current_raw_match'],1)
        self.assertEqual(audit['scored_current_matched_but_available_later'],1)
        self.assertEqual(audit['scored_future_raw_match'],1)


if __name__=='__main__': unittest.main()
