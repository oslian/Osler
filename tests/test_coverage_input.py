import unittest

import numpy as np
import pandas as pd
import torch

from scripts.encoder_fold_scaling import transform_example
from whole_body_jepa.coverage_input import build_input
from whole_body_jepa.native_reconstruction_examples import reconstruction_examples


class CoverageInputTests(unittest.TestCase):
    def checkpoint(self):
        return dict(format='masked-current-reconstruction-coverage-v1', variables=['map', 'glucose'],
            targets=['map', 'glucose'], schema=dict(modules=['cv', 'endo'],
                state_dict={'variable_module_ids': torch.tensor([0, 1])}),
            scaler=dict(location=torch.tensor([80., 100.]), scale=torch.tensor([10., 20.]),
                        counts=torch.tensor([100, 100])))

    def events(self):
        return pd.DataFrame([
            dict(stay_id='s', source_kind='lab_available', source_id='g', event_time='2100-01-01 00:00:00',
                 measured_at='2100-01-01 00:00:00', variable='glucose', value=0., unit='mg/dL', quality_valid=True),
            dict(stay_id='s', source_kind='bedside_available', source_id='m', event_time='2100-01-01 01:00:00',
                 measured_at='2100-01-01 01:00:00', variable='map', value=80., unit='mmHg', quality_valid=True)])

    def build(self, events):
        return build_input(events, self.checkpoint(), 's', '2100-01-01 01:00:00', 'map')

    def test_matches_original_training_builder_without_needing_target_truth(self):
        events = self.events()
        actual = self.build(events)
        missing_target = self.build(events.iloc[:1])
        reference = list(reconstruction_examples(events, pd.DataFrame([dict(stay_id='s', t='2100-01-01 01:00:00')]),
            ['map', 'glucose'], np.zeros(2), np.ones(2), np.array([0, 1]), 2))[0]
        scaler = {k: v.numpy() for k, v in self.checkpoint()['scaler'].items()}
        reference = transform_example(reference, scaler)
        for other in (reference, missing_target):
            torch.testing.assert_close(actual['tokens'], other['tokens'], rtol=0, atol=0)
            torch.testing.assert_close(actual['event_hours'], other['event_hours'], rtol=0, atol=0)
            self.assertEqual(actual['anchor_hour'], other['anchor_hour'])
        self.assertEqual(float(actual['tokens'][0, 3]), 1.)
        self.assertEqual(float(actual['tokens'][0, 1]), -5.)

    def test_future_and_other_stay_do_not_change_input(self):
        baseline = self.build(self.events())
        extra = self.events().iloc[[0]].copy()
        extra['event_time'] = '2100-01-01 02:00:00'
        extra['value'] = 999.
        other = self.events().copy()
        other['stay_id'] = 'other'
        actual = self.build(pd.concat([self.events(), extra, other]))
        torch.testing.assert_close(actual['tokens'], baseline['tokens'], rtol=0, atol=0)

    def test_units_missing_time_and_treatment_rejected(self):
        for column, value in [('unit', 'mmol/L'), ('measured_at', None),
                              ('source_kind', 'insulin_record'), ('value', np.nan)]:
            events = self.events()
            events.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.build(events)

    def test_stale_and_no_independent_data_fail_closed(self):
        events = self.events()
        events.loc[0, 'event_time'] = '2100-01-01 01:00:00'
        events.loc[0, 'measured_at'] = '2099-12-31 22:00:00'
        with self.assertRaisesRegex(ValueError, 'no independent observed history'):
            self.build(events)
        with self.assertRaises(ValueError):
            self.build(self.events().iloc[1:])


if __name__ == '__main__':
    unittest.main()
