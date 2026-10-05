import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples


class NativeExamplesTest(unittest.TestCase):
    def events(self):
        origin=pd.Timestamp('2100-01-01')
        rows=[]
        for source,var,measured,available,value in [('g','glucose',0,0,100),('m','map',1,1,80),('m','map',1,2,90)]:
            rows.append(dict(stay_id='s',source_kind='lab_available',source_id=source,
                variable=var,measured_at=origin+pd.Timedelta(hours=measured),
                event_time=origin+pd.Timedelta(hours=available),value=value,quality_valid=True))
        return pd.DataFrame(rows)

    def build(self,events,times=(1,2)):
        q=pd.DataFrame(dict(stay_id=['s']*len(times),t=[pd.Timestamp('2100-01-01')+pd.Timedelta(hours=t) for t in times]))
        return list(reconstruction_examples(events,q,('map','glucose'),np.zeros(2),np.ones(2),np.array([0,1]),2,max_age_hours=6))

    def test_future_revision_does_not_rewrite_earlier_truth(self):
        all_rows=self.build(self.events())
        earlier=self.build(self.events().iloc[:2],(1,))
        self.assertEqual([x['truth_normalized'] for x in all_rows],[80,90])
        self.assertEqual(earlier[0]['truth_normalized'],all_rows[0]['truth_normalized'])
        for row in all_rows:
            self.assertTrue((row['tokens'][:,2]==0).all())
            self.assertTrue((row['event_hours']<=row['anchor_hour']).all())

    def test_unknown_or_stale_anchor_is_not_a_label(self):
        self.assertEqual(self.build(self.events(),(3,)),[])

    def test_encounter_isolation(self):
        e=self.events();e.loc[e.variable=='glucose','stay_id']='other'
        self.assertEqual(self.build(e),[])

    def test_conflicting_known_measurement_is_not_truth(self):
        e=self.events();extra=e.iloc[[1]].copy();extra['source_id']='conflict';extra['value']=120
        self.assertEqual(self.build(pd.concat([e,extra],ignore_index=True),(1,)),[])

    def test_example_runs_through_shared_body_and_backprop(self):
        import torch
        from whole_body_joint_jepa import JointWholeBodyJEPAWorldModel
        from whole_body_jepa.history_event_input import expand_history_projection
        from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual
        from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
        torch.manual_seed(7)
        j=JointWholeBodyJEPAWorldModel(2,10,2,hidden=12,latent=8,token_dim=8,
            variable_module_ids=[0,1],horizon_values=(3.,),target_encoder_mode='fast')
        expand_history_projection(j,6);j.belief_input_mode='observed_steps'
        model=NativeBodyReconstruction(NativeOrganStateResidual(j,('map','glucose'),('map',)))
        e=self.build(self.events(),(1,))[0]
        self.assertEqual(e['tokens'].shape[1],j.input_dim)
        value=model(e['tokens'],e['event_hours'],e['anchor_hour'],'map')
        (value-e['truth_normalized']).square().sum().backward()
        self.assertTrue(torch.isfinite(value).all())
        self.assertGreater(sum(float(p.grad.square().sum()) for p in model.trajectory.layer.body_update_gate.parameters() if p.grad is not None),0)
