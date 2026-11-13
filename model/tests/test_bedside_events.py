import unittest

import numpy as np
import pandas as pd

from whole_body_jepa.bedside_events import nursing_events
from whole_body_jepa.availability_inputs import snapshots
from whole_body_jepa.native_history_schedule import conflict_checked_history_schedule
from whole_body_jepa.native_event_tokens import encode_group


def row(identity='1', **changes):
    result = dict(nursingchartid=identity, patientunitstayid='9', nursingchartoffset='0',
                  nursingchartentryoffset='10', nursingchartvalue='80',
                  nursingchartcelltypecat='Vital Signs',
                  nursingchartcelltypevallabel='Heart Rate', nursingchartcelltypevalname='Heart Rate')
    result.update(changes)
    return result


class BedsideEventsTests(unittest.TestCase):
    def test_invalid_revision_does_not_restore_earlier_version(self):
        events = nursing_events(pd.DataFrame([row(), row(nursingchartentryoffset='20', nursingchartvalue='0')]))
        events['quality_valid'] = [True, False]
        events['event_time'] = events.available_at
        queries = pd.DataFrame(dict(stay_id=['eicu:9'] * 2,
            t=pd.to_datetime(['2100-01-01 00:10', '2100-01-01 00:20'])))
        canonical, schedules = conflict_checked_history_schedule(events, queries)
        self.assertEqual(canonical.loc[schedules[0][0], 'value'].tolist(), [80.])
        self.assertEqual(canonical.loc[schedules[1][0], 'value'].tolist(), [0.])
        token, _ = encode_group(canonical.loc[schedules[1][0]], ['heart_rate'], [70.], [10.], [0], 1, 6.)
        self.assertEqual(token[1], 0.)
        values, _ = snapshots(events, queries.t, ['heart_rate'], 6.)
        self.assertEqual(values[0, 0], 80.)
        self.assertTrue(np.isnan(values[1, 0]))

    def test_invalid_latest_does_not_resurrect_old_value(self):
        events = nursing_events(pd.DataFrame([row(), row('2', nursingchartoffset='20',
            nursingchartentryoffset='20', nursingchartvalue='0')]))
        events['quality_valid'] = [True, False]
        values, _ = snapshots(events, [pd.Timestamp('2100-01-01 00:20')], ['heart_rate'], 6.)
        self.assertTrue(np.isnan(values[0, 0]))
        events['event_time'] = events.available_at
        token, _ = encode_group(events.iloc[[1]], ['heart_rate'], [70.], [10.], [0], 1, 6.)
        self.assertEqual(token[1], 0.)

    def test_raw_nursing_to_native_encoder(self):
        events = nursing_events(pd.DataFrame([row()]))
        events['event_time'] = events.available_at
        queries = pd.DataFrame(dict(stay_id=['eicu:9'] * 2,
            t=pd.to_datetime(['2100-01-01 00:05', '2100-01-01 00:10'])))
        canonical, schedules = conflict_checked_history_schedule(events, queries)
        self.assertEqual(len(schedules[0]), 0)
        token, _ = encode_group(canonical.loc[schedules[1][0]], ['heart_rate'],
                                [70.], [10.], [0], 1, 6.)
        self.assertEqual(token[0], 1.)
        self.assertEqual(token[1], 1.)
        self.assertAlmostEqual(float(token[2]), 10 / 60 / 24)

    def test_zero_remeasurement_and_dedup(self):
        first = row(nursingchartvalue='0')
        events = nursing_events(pd.DataFrame([first, first, row('2', nursingchartvalue='0')]))
        self.assertEqual(len(events), 2)
        self.assertTrue(events.value.eq(0).all())
        self.assertTrue(events.source_id.str.startswith('nurse:').all())

    def test_availability_and_invalid_records(self):
        events = nursing_events(pd.DataFrame([
            row(), row('2', nursingchartentryoffset='-1'), row('3', nursingchartvalue='inf'),
            row(None), row('4', nursingchartoffset='nan')]))
        self.assertEqual(len(events), 1)
        times = pd.to_datetime(['2100-01-01 00:05', '2100-01-01 00:10'])
        values, _ = snapshots(events, times, ['heart_rate'], 6)
        self.assertTrue(np.isnan(values[0, 0]))
        self.assertEqual(values[1, 0], 80)

    def test_temperature_and_ambiguous_oxygen(self):
        events = nursing_events(pd.DataFrame([
            row('1', nursingchartcelltypevallabel='Temperature', nursingchartcelltypevalname='Temperature (C)', nursingchartvalue='37'),
            row('2', nursingchartcelltypevallabel='Temperature', nursingchartcelltypevalname='Temperature (F)', nursingchartvalue='98.6'),
            row('3', nursingchartcelltypevallabel='O2 L/%', nursingchartcelltypevalname='O2 L/%')]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0].unit_basis, 'explicit_label')

    def test_map_conflict_not_silently_selected(self):
        events = nursing_events(pd.DataFrame([
            row('1', nursingchartcelltypevallabel='Invasive BP', nursingchartcelltypevalname='Invasive BP Mean'),
            row('2', nursingchartcelltypevallabel='Non-Invasive BP', nursingchartcelltypevalname='Non-Invasive BP Mean', nursingchartvalue='100')]))
        self.assertEqual(events.measurement_method.nunique(), 2)
        values, _ = snapshots(events, [pd.Timestamp('2100-01-01 00:10')], ['map'], 6)
        self.assertTrue(np.isnan(values[0, 0]))


if __name__ == '__main__':
    unittest.main()
