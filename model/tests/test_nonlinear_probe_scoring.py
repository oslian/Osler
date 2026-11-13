import unittest

import numpy as np
import pandas as pd

from scripts.score_nonlinear_probe import ARMS, assess_target, patient_errors, verify_probe_frame


def example():
    frame = pd.DataFrame(dict(patient=['a', 'b', 'c'], component=['x', 'y', 'z'],
                              fold=0, target='map', truth=[1., 2., 3.]))
    for arm in ARMS:
        frame[arm] = [0., 2., 4.]
    return frame


class ProbeScoringTests(unittest.TestCase):
    def test_overlap_is_loss_average_not_new_patients_or_prediction_ensemble(self):
        first = example()
        second = first.copy()
        second['fold'] = 1
        second['native'] = [2., 2., 2.]
        merged = pd.concat([first, second], ignore_index=True)
        error = patient_errors(merged)
        self.assertEqual(len(error), 3)
        np.testing.assert_array_equal(error.native, [1., 0., 1.])

    def test_duplicate_or_changed_component_rejected(self):
        first = example()
        with self.assertRaises(ValueError):
            patient_errors(pd.concat([first, first], ignore_index=True))
        second = first.copy()
        second['fold'] = 1
        second.loc[0, 'component'] = 'foreign'
        with self.assertRaises(ValueError):
            patient_errors(pd.concat([first, second], ignore_index=True))

    def test_contract_checks_roles_grid_and_scale(self):
        frame = example()
        base = frame[['patient', 'truth', 'population']].copy()
        roles = dict.fromkeys(frame.patient, 'selection')
        components = dict(zip(frame.patient, frame.component))
        verify_probe_frame(frame, base, roles, 0, 'map', 0., 1., components)
        with self.assertRaises(ValueError):
            verify_probe_frame(frame.iloc[:2], base, roles, 0, 'map', 0., 1., components)
        with self.assertRaises(ValueError):
            verify_probe_frame(frame, base, {**roles, 'a': 'fit'}, 0, 'map', 0., 1., components)
        with self.assertRaises(ValueError):
            verify_probe_frame(frame, base, roles, 0, 'map', 0., 2., components)

    def test_identical_predictions_no_pass_and_small_support_no_pass(self):
        result = assess_target(example(), 19, repetitions=50)
        self.assertEqual(result['unique_patients'], 3)
        for comparison in result['comparisons'].values():
            self.assertFalse(comparison['per_contrast_47_screen'])
            self.assertFalse(comparison['all_8_contrasts_47_screen'])
            self.assertEqual(comparison['paired']['patient']['mean'], 0.)


if __name__ == '__main__':
    unittest.main()
