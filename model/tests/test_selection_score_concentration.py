import unittest
import numpy as np
from scripts.audit_selection_score_concentration import concentration


def arm(initial, later, selected=1):
    return dict(initial_scores={'group': dict(zip(('x', 'y'), initial))},
                selected_epoch=selected, log=[dict(epoch=1, tune_mae=float(np.mean(later)),
                group_target_scores={'group': dict(zip(('x', 'y'), later))})])


class ConcentrationTest(unittest.TestCase):
    def test_exact_contributions_and_scale_tradeoff(self):
        result = concentration(arm([100., 1.], [90., 2.]))
        rows = result['per_target']
        self.assertAlmostEqual(sum(x['selected_score_contribution'] for x in rows.values()), 46.)
        self.assertAlmostEqual(sum(x['initial_to_selected_gain_contribution'] for x in rows.values()), 4.5)
        self.assertEqual(rows['x']['leave_one_target_out_epoch'], 0)
        self.assertEqual(result['reference_relative_best_epoch'], 0)
        self.assertEqual(result['selected_harms_initial_count'], 1)

    def test_recorded_selection_rejected(self):
        with self.assertRaisesRegex(ValueError, 'selection'):
            concentration(arm([1., 1.], [2., 2.]))

    def test_zero_reference_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Positive'):
            concentration(arm([0., 1.], [.1, .5]))
