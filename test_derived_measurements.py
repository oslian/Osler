import unittest

import numpy as np

from osler_jepa.derived_measurements import calculated_bicarbonate, mean_arterial_pressure


class DerivedMeasurementTests(unittest.TestCase):
    def test_bicarbonate_matches_textbook_values(self):
        # normal arterial gas: pH 7.40, PaCO2 40 -> about 24 mmol/L
        self.assertAlmostEqual(float(calculated_bicarbonate(7.40, 40.0)), 24.5, delta=0.3)
        # metabolic acidosis: pH 7.20, PaCO2 25 -> about 9.6 mmol/L
        self.assertAlmostEqual(float(calculated_bicarbonate(7.20, 25.0)), 9.6, delta=0.3)
        self.assertTrue(np.isnan(calculated_bicarbonate(np.nan, 40.0)))

    def test_map_requires_systolic_above_diastolic(self):
        np.testing.assert_allclose(mean_arterial_pressure([120.0, 80.0], [80.0, 90.0]), [280.0 / 3.0, np.nan])


if __name__ == "__main__":
    unittest.main()
