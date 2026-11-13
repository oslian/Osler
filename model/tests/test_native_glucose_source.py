import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.native_glucose_source import source_indicator


class NativeGlucoseSourceTest(unittest.TestCase):
    def group(self, names):
        return pd.DataFrame(dict(source_kind=['lab_available'] * len(names),
                                 variable=['glucose'] * len(names),
                                 measured_at=pd.to_datetime(['2100-01-01'] * len(names)),
                                 source_test_name=names))

    def test_each_source(self):
        for name, expected in [('bedside glucose', [1, 0]), ('glucose', [0, 1])]:
            np.testing.assert_array_equal(source_indicator(self.group([name]), True), expected)

    def test_unobserved_and_unknown(self):
        for names in [['unknown'], [None], ['glucose', None], ['glucose', 'bedside glucose']]:
            np.testing.assert_array_equal(source_indicator(self.group(names), True), [0, 0])
        np.testing.assert_array_equal(source_indicator(self.group(['glucose']), False), [0, 0])

    def test_latest_measurement_only(self):
        group = self.group(['bedside glucose', 'glucose'])
        group.loc[1, 'measured_at'] += pd.Timedelta(hours=1)
        np.testing.assert_array_equal(source_indicator(group, True), [0, 1])

    def test_repeated_same_source_and_nonlab(self):
        np.testing.assert_array_equal(source_indicator(self.group(['glucose', 'glucose']), True), [0, 1])
        group = self.group(['glucose'])
        group['source_kind'] = 'insulin_record'
        with self.assertRaises(ValueError):
            source_indicator(group, True)
