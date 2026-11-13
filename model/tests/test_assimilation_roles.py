import unittest

import pandas as pd

from scripts.audit_assimilation_roles import audit_roles


def frame(rows):
    return pd.DataFrame(rows, columns=['patient', 'component', 'variable', 'fold', 'context'])


class AssimilationRolesTest(unittest.TestCase):
    def test_pooled_overlap_cannot_create_within_model_control(self):
        fit = frame([('a', 'x', 'v', 0, 0), ('b', 'y', 'v', 1, 0)])
        selection = frame([('b', 'y', 'v', 0, 0), ('a', 'x', 'v', 1, 0)])
        result = audit_roles(fit, selection, ['v', 'unsupported'])
        self.assertEqual(result['summary']['folds_with_common_components'], 0)
        self.assertEqual(result['summary']['pooled_component_overlap_not_within_model'], 2)
        self.assertEqual(result['summary']['target_fold_status_counts'], dict(
            missing_role_support=2, same_source_support_exists=0,
            same_source_comparison_unavailable=2))

    def test_same_source_different_patients_available(self):
        fit = frame([('a', 'x', 'v', 0, 0)])
        selection = frame([('b', 'x', 'v', 0, 0), ('c', 'y', 'v', 0, 0)])
        result = audit_roles(fit, selection, ['v'])['folds']['0']
        self.assertEqual(result['common_components'], 1)
        self.assertEqual(result['selection_common_occurrences'], 1)
        self.assertEqual(result['targets']['v']['comparison_status'], 'same_source_support_exists')

    def test_target_support_not_inherited_from_other_variables(self):
        fit = frame([('a', 'x', 'v', 0, 0), ('b', 'y', 'w', 0, 1)])
        selection = frame([('c', 'y', 'v', 0, 0), ('d', 'x', 'w', 0, 1)])
        result = audit_roles(fit, selection, ['v', 'w'])['folds']['0']
        self.assertEqual(result['common_components'], 2)
        self.assertTrue(all(t['common_components'] == 0 for t in result['targets'].values()))

    def test_reject_patient_leakage_and_provenance_change(self):
        fit = frame([('a', 'x', 'v', 0, 0)])
        for selection in (frame([('a', 'x', 'v', 0, 0)]),
                          frame([('a', 'y', 'v', 0, 0)])):
            with self.assertRaises(ValueError): audit_roles(fit, selection, ['v'])

    def test_missing_duplicate_unknown_and_context_alias_rejected(self):
        fit = frame([('a', 'x', 'v', 0, 0)])
        bad_frames = [frame([(None, 'y', 'v', 0, 0)]),
            frame([('b', 'y', 'v', 0, 0), ('b', 'y', 'v', 0, 0)]),
            frame([('b', 'y', 'unknown', 0, 0)]),
            frame([('b', 'y', 'v', 0, 1)]),
            frame([('b', 'y', 'w', 0, 0)])]
        for bad in bad_frames:
            with self.assertRaises(ValueError): audit_roles(fit, bad, ['v', 'w'])


if __name__ == '__main__':
    unittest.main()
