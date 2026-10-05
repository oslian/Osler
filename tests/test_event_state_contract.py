import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from scripts.probe_event_state_contract import run, replay
from scripts.audit_event_aware_patient_state import replay_features, _treatment_matrix, measurement_matrix
from whole_body_jepa.event_treatment import rate_value, snapshot_summary
from whole_body_jepa.recurrent_event_state import (
    RecurrentSignedResidual, FeatureStandardizer, feedback_sequence,
    _padded_batch, fit_recurrent_signed_residual,
    predict_recurrent,
)


class EventContractTests(unittest.TestCase):
    def test_three_reproductions_are_fixed(self):
        run()

    def test_carried_value_not_a_new_measurement(self):
        frame = pd.DataFrame(dict(subject_id=['a','a'], stay_id=['a','a'],
            t=pd.date_range('2020-01-01', periods=2, freq='1h'), _time_hours=[0.,1.],
            careunit=['test','test'], future_event_time_glucose=[pd.NaT,pd.NaT],
            future_delta_t_hr_glucose=[np.nan,np.nan]))
        prepared = SimpleNamespace(frame=frame,current=np.array([100.,100.]),
            truth=np.array([np.nan,np.nan]),base=np.array([100.,100.]))
        records, x = replay_features(prepared, (np.ones((2,1)),np.ones((2,1)),np.array([[0.],[1/6]])),
                                    np.zeros((2,10)), 100., reference_center=100.,max_age_hours=6.)
        self.assertEqual(records.iloc[1].new_measurement_fraction, 0)
        self.assertEqual(len(x), 2)  # Missing future labels do not drop updates.
        self.assertTrue(np.isfinite(x).all())

    def test_label_mask_does_not_drop_update_features(self):
        x, y, mask = _padded_batch([np.array([0,1,2])],np.arange(9,dtype=np.float32).reshape(3,3),
                                   np.array([1,np.nan,2],dtype=np.float32))
        self.assertEqual(x.shape[1],3)
        self.assertEqual(mask.tolist(),[[1,0,1]])
        self.assertTrue(torch.isfinite(y).all())

    def test_stale_age_is_not_clipped_into_a_fake_new_event(self):
        frame = pd.DataFrame({'glucose_t':[100.,100.,100.], 'glucose_age_hr':[10.,11.,np.nan]})
        assets = SimpleNamespace(checkpoint={'variables':['glucose']},
            scaler=SimpleNamespace(medians=[100.],scales=[100.]),max_age_hours=6.)
        _, observed, ages = measurement_matrix(frame,assets)
        self.assertEqual(ages.dtype,np.float64)
        np.testing.assert_allclose(ages[:2,0],[10/6,11/6])
        times = np.array([0.,1.])-ages[:2,0]*6
        self.assertAlmostEqual(float(times[0]),float(times[1]),places=5)
        self.assertEqual(observed[2,0],0.)

    def test_rate_units_and_zero(self):
        self.assertEqual(rate_value(dict(rate_like=0,rate_unit='units/hr')), (0.,'drug_units_per_hour'))
        self.assertEqual(rate_value(dict(rate_like=2,source='eicu_infusiondrug',rate_field='drugrate',original_label='Insulin (units/hr)')), (2.,'drug_units_per_hour'))
        self.assertIsNone(rate_value(dict(rate_like=2,source='eicu_infusiondrug',rate_field='infusionrate',original_label='Insulin (units/hr)'))[1])
        self.assertIsNone(rate_value(dict(rate_like=2,source='eicu_infusiondrug',original_label='Insulin'))[1])
        self.assertIsNone(rate_value(dict(rate_like=2,rate_unit='unknown'))[1])

    def test_zero_rate_distinct_from_unknown(self):
        missing = _treatment_matrix(pd.DataFrame(index=[0]))
        zero = _treatment_matrix(pd.DataFrame({'hist_insulin_last_rate_drug_units_per_hour':[0.]}))
        self.assertFalse(np.array_equal(missing,zero))

    def test_snapshot_cannot_use_future_rate_or_unfinished_amount(self):
        start = pd.Timestamp('2020-01-01')
        anchors = pd.DataFrame(dict(stay_id=[1],t=[start+pd.Timedelta(hours=1)]))
        evidence = pd.DataFrame([dict(stay_id=1,starttime=start,endtime=start+pd.Timedelta(hours=4),
            action='insulin',source='mimic_inputevents',rate_like=1.,rate_unit='units/hr',
            amount_like=4.,amount_unit='units',original_label='Insulin'),
            dict(stay_id=1,starttime=start+pd.Timedelta(hours=2),endtime=start+pd.Timedelta(hours=4),
            action='insulin',source='mimic_inputevents',rate_like=10.,rate_unit='units/hr',
            amount_like=20.,amount_unit='units',original_label='Insulin')])
        summary = snapshot_summary(anchors,evidence)
        self.assertEqual(summary.iloc[0].hist_insulin_last_rate_drug_units_per_hour,1.)
        self.assertTrue(pd.isna(summary.iloc[0].hist_insulin_dose_drug_units))

    def test_feedback_waits_for_outcome_and_uses_own_prediction(self):
        records, features = replay([100.]*8)
        model = RecurrentSignedResidual(features.shape[1],hidden_width=4)
        with torch.no_grad():
            model.readout[-1].bias.fill_(0.1)
        scaler = FeatureStandardizer(np.zeros(features.shape[1],np.float32),np.ones(features.shape[1],np.float32))
        inputs = []
        hook = model.event_encoder.register_forward_pre_hook(lambda m,args: inputs.append(args[0].detach().clone()))
        prediction = feedback_sequence(model,scaler,records,features,100.)
        hook.remove()
        self.assertEqual(float(inputs[5][0,0,0]),0.)
        expected = (10. - float(prediction[0].detach())*100.) / 4. / 100.
        self.assertAlmostEqual(float(inputs[6][0,0,0]),expected,places=6)
        changed = records.copy()
        changed['truth'] += 500
        altered = feedback_sequence(model,scaler,changed,features,100.)
        torch.testing.assert_close(prediction[:6],altered[:6])

    def test_zero_checkpoint_available_without_training(self):
        frame = pd.DataFrame(dict(subject_id=[str(i) for i in range(60)],
            t=[pd.Timestamp('2020-01-01')]*60,_time_hours=np.zeros(60)))
        fit = np.arange(60)<50
        _, model, audit = fit_recurrent_signed_residual(frame,np.ones((60,2),np.float32),fit,~fit,
            truth=np.zeros(60),frozen=np.zeros(60),residual_scale=1.,seed=1,epochs=0)
        self.assertEqual(audit['selected_epoch'],-1)
        self.assertEqual(float(model(torch.ones(1,1,2)).detach()),0.)

    def test_online_predictions_are_independent_of_unrelated_patients(self):
        frame, features = replay([100.]*8)
        other = frame.copy()
        other['subject_id'] = 'other'
        other['truth'] += 1000
        combined = pd.concat([frame,other],ignore_index=True)
        x = np.concatenate([features,features+3])
        torch.manual_seed(5)
        model = RecurrentSignedResidual(features.shape[1],hidden_width=4)
        with torch.no_grad():
            model.readout[-1].weight.fill_(0.1)
        scaler = FeatureStandardizer(np.zeros(features.shape[1],np.float32),np.ones(features.shape[1],np.float32))
        alone = predict_recurrent(model,scaler,frame,features,feedback_scale=100.)
        together = predict_recurrent(model,scaler,combined,x,feedback_scale=100.)
        np.testing.assert_array_equal(alone,together[:8])


if __name__ == '__main__':
    unittest.main()
