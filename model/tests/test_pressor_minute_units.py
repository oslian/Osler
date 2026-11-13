import unittest
import math
from osler_jepa.treatment_units import canonical_rate
from whole_body_jepa.event_treatment import rate_channels


class PressorMinuteUnits(unittest.TestCase):
    def test_minute_rate_has_exact_hourly_dimension(self):
        for unit in ('unit/min', 'units/min', 'unit/minute', 'units/minute'):
            value, dimension = canonical_rate(.03, unit)
            self.assertEqual(dimension, 'drug_units_per_hour')
            self.assertAlmostEqual(value, 1.8)

    def test_documented_zero_remains_observed_zero(self):
        result = rate_channels(dict(source='eicu_infusiondrug', rate_field='drugrate',
            rate_like=0, original_label='Vasopressin (units/min)'), 'vasopressor')
        self.assertEqual(result['action_rate'], 0)

    def test_solution_flow_does_not_become_drug_rate(self):
        result = rate_channels(dict(source='eicu_infusiondrug', rate_field='drugrate',
            rate_like=3, original_label='Vasopressin (ml/hr)'), 'vasopressor')
        self.assertTrue(math.isnan(result['action_rate']))
        self.assertEqual(result['solution_flow_ml_per_hour'], 3)


if __name__ == '__main__': unittest.main()
