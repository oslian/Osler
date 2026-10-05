import unittest

from scripts.diagnose_correction_concentration import summarize_components


class ConcentrationTest(unittest.TestCase):
    def test_single_component_can_drive_mean(self):
        rows = [dict(mae=x, reference_mae={'unchanged': 10}, patients=100) for x in (1, 11, 11)]
        result = summarize_components(rows)
        self.assertLess(result['mean_delta'], 0)
        self.assertFalse(result['majority_improves'])
        self.assertGreater(result['worst_leave_one_out_delta'], 0)

    def test_small_components_not_silently_removed(self):
        rows = [dict(mae=1, reference_mae={'unchanged': 10}, patients=1),
                dict(mae=11, reference_mae={'unchanged': 10}, patients=100)]
        result = summarize_components(rows)
        self.assertLess(result['mean_delta'], 0)
        self.assertEqual(result['supported_only_delta'], 1)
        self.assertEqual(result['components'], 2)
