import unittest
import numpy as np
import pandas as pd
from whole_body_jepa.recorded_pressor_context import recorded_snapshots


class RecordedPressors(unittest.TestCase):
    def events(self, rows):
        return pd.DataFrame(rows, columns=['stay', 'id', 'drug', 'dimension', 'offset', 'rate'])

    def test_zero_future_and_prefix(self):
        rows = [(1, 1, 'vasopressin', 'drug_units_per_hour', 10, 0),
                (1, 2, 'vasopressin', 'drug_units_per_hour', 30, 3)]
        anchors = pd.DataFrame({'stay': [1], 'offset': [20]})
        a = recorded_snapshots(anchors, self.events(rows[:1]))
        b = recorded_snapshots(anchors, self.events(rows))
        np.testing.assert_array_equal(a['rate'], b['rate'])
        self.assertEqual(b['rate'][0, 4, 2], 0)
        self.assertFalse(b['online_eligible'])

    def test_invalid_latest_blocks_old_valid(self):
        rows = [(1, 1, 'vasopressin', 'drug_units_per_hour', 10, 2),
                (1, 2, 'vasopressin', None, 20, np.nan)]
        out = recorded_snapshots(pd.DataFrame({'stay': [1], 'offset': [25]}), self.events(rows))
        self.assertFalse(out['known'].any())
        self.assertEqual(out['status'][0, 4], 'invalid_latest')

    def test_conflict_not_sum_and_no_cross_patient(self):
        rows = [(1, 1, 'vasopressin', 'drug_units_per_hour', 10, 2),
                (1, 2, 'vasopressin', 'drug_units_per_hour', 10, 3)]
        out = recorded_snapshots(pd.DataFrame({'stay': [1, 2], 'offset': [20, 20]}), self.events(rows))
        self.assertFalse(out['known'].any())
        self.assertEqual(out['status'][0, 4], 'conflicting_latest')
        self.assertEqual(out['status'][1, 4], 'missing')

    def test_exact_duplicates_inert_remeasurements_preserved(self):
        rows = [(1, 1, 'vasopressin', 'drug_units_per_hour', 10, 2)] * 2
        rows.append((1, 2, 'vasopressin', 'drug_units_per_hour', 20, 2))
        out = recorded_snapshots(pd.DataFrame({'stay': [1], 'offset': [25]}), self.events(rows))
        self.assertAlmostEqual(out['age_hours'][0, 4], 5 / 60)

    def test_source_conflict_rejected(self):
        rows = [(1, 1, 'vasopressin', 'drug_units_per_hour', 10, 2),
                (1, 1, 'vasopressin', 'drug_units_per_hour', 10, 3)]
        with self.assertRaises(ValueError):
            recorded_snapshots(pd.DataFrame({'stay': [1], 'offset': [20]}), self.events(rows))


if __name__ == '__main__': unittest.main()
