import unittest
import pandas as pd

from scripts.extract_additional_domain_bedside import canonical_bedside
from scripts.extract_additional_domain_labs import fresh_unambiguous


class AdditionalBedsideTests(unittest.TestCase):
    def raw(self):
        return pd.DataFrame(dict(nursingchartid=['1', '2'], patientunitstayid=['10']*2,
            nursingchartoffset=['0', '30'], nursingchartentryoffset=['5', '35'],
            nursingchartcelltypecat=['Vital Signs']*2,
            nursingchartcelltypevallabel=['Heart Rate']*2,
            nursingchartcelltypevalname=['Heart Rate']*2, nursingchartvalue=['80', '80']))

    def provenance(self):
        return pd.DataFrame(dict(stay_id=['eicu:10'], uniquepid=['person'],
            hospitalid=['site'], component=['domain']))

    def test_equal_remeasurements_and_entry_time_preserved(self):
        events = canonical_bedside(self.raw(), self.provenance())
        self.assertEqual(len(events), 2)
        self.assertEqual(events.source_id.nunique(), 2)
        self.assertTrue(events.quality_valid.all())
        self.assertTrue(events.event_time.eq(events.available_at).all())
        self.assertTrue(((events.available_at-events.measured_at).dt.total_seconds()==300).all())

    def test_invalid_zero_not_imputed_or_valid_label(self):
        raw = self.raw(); raw.loc[1, 'nursingchartvalue'] = '0'
        events = canonical_bedside(raw, self.provenance())
        self.assertEqual(events.value.tolist(), [80., 0.])
        self.assertEqual(events.quality_valid.tolist(), [True, False])

    def test_impossible_entry_time_rejected(self):
        raw = self.raw(); raw.loc[1, 'nursingchartentryoffset'] = '29'
        self.assertEqual(len(canonical_bedside(raw, self.provenance())), 1)

    def test_latest_invalid_does_not_rescue_older_draw(self):
        raw = self.raw(); raw['nursingchartentryoffset'] = '35'
        raw.loc[1, 'nursingchartvalue'] = '0'
        eligible = fresh_unambiguous(canonical_bedside(raw, self.provenance()))
        self.assertEqual(eligible.value.tolist(), [0.])
        self.assertFalse(eligible.quality_valid.any())

    def test_missing_provenance_rejected(self):
        provenance = self.provenance(); provenance['stay_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            canonical_bedside(self.raw(), provenance)

    def test_empty_hospital_retains_canonical_schema(self):
        events = canonical_bedside(self.raw().iloc[:0], self.provenance())
        self.assertTrue(events.empty)
        self.assertIn('available_at', events)
        self.assertIn('quality_valid', events)
        self.assertTrue(fresh_unambiguous(events).empty)
