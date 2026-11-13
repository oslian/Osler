import unittest
import numpy as np
from whole_body_jepa.event_treatment import rate_channels


class RateChannelTests(unittest.TestCase):
    def record(self,label,value,field='drugrate'):
        return dict(source='eicu_infusiondrug',original_label=label,rate_like=value,rate_field=field)

    def test_volume_is_not_potassium_dose(self):
        r=rate_channels(self.record('KCl (ml/hr)',25),'potassium_repletion')
        self.assertEqual(r['solution_flow_ml_per_hour'],25)
        self.assertTrue(np.isnan(r['action_rate']))
        self.assertIsNone(r['action_rate_dimension'])

    def test_known_zero_and_charge(self):
        r=rate_channels(self.record('KCl (mEq/hr)',0),'potassium_repletion')
        self.assertEqual(r['action_rate'],0)
        self.assertEqual(r['action_rate_dimension'],'charge_meq_per_hour')
        self.assertTrue(np.isnan(r['solution_flow_ml_per_hour']))

    def test_no_inherited_or_ambiguous_units(self):
        for record in (self.record('KCl (mEq/hr)',25,'infusionrate'),
                       self.record('KCl (Unknown)',25),
                       self.record('KCl (mEq/hr) (ml/hr)',25)):
            r=rate_channels(record,'potassium_repletion')
            self.assertTrue(np.isnan(r['action_rate']))
            self.assertTrue(np.isnan(r['solution_flow_ml_per_hour']))
