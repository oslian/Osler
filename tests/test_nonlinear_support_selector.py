import unittest

import numpy as np
import pandas as pd

from scripts.probe_nonlinear_support_selector import support_features, crossfit, assess


def example():
    x = np.tile(np.linspace(-1,1,40),3)
    f = pd.DataFrame(dict(patient=[f'p{i}' for i in range(120)],component=np.repeat(['a','b','c'],40),
        fold=0,target='t',role='selection',truth=0.,native=1.,body_probe=1+.6*x,population=1.))
    aggregate = np.column_stack((x,np.zeros((len(x),3))))
    variable = np.column_stack((x,np.zeros((len(x),105))))
    return f,aggregate,variable


class SupportSelectorTests(unittest.TestCase):
    def test_missing_age_excluded_and_target_mask_required(self):
        f = pd.DataFrame(dict(observed_t=[False],observed_a=[True],observed_b=[False],age_t=[np.nan],age_a=[2.],age_b=[np.nan]))
        aggregate,variable = support_features(f,['t','a','b'],'t')
        np.testing.assert_allclose(aggregate,[[1,0,1,2]])
        np.testing.assert_allclose(variable,[[0,1,0,0,np.log(3),0]])
        f['observed_t'] = True; f['age_t'] = 0.
        with self.assertRaises(ValueError): support_features(f,['t','a','b'],'t')

    def test_held_component_outcomes_cannot_change_own_decisions(self):
        f,a,v = example()
        first,_ = crossfit(f,a,v)
        changed = f.copy(); changed.loc[changed.component=='a','truth'] = 100.
        second,_ = crossfit(changed,a,v)
        for name in ('constant','aggregate','variable','shuffled'):
            np.testing.assert_allclose(first.loc[f.component=='a','expected_'+name],second.loc[f.component=='a','expected_'+name])

    def test_recovers_a_transportable_decision_and_patient_weighting(self):
        f,a,v = example()
        predicted,groups = crossfit(f,a,v)
        self.assertTrue(predicted.supported.all())
        self.assertEqual(len(groups),3)
        result = assess(predicted,1,repetitions=30)
        self.assertLess(result['normalized_mae']['patient']['variable'],result['normalized_mae']['patient']['native'])
        self.assertLess(result['normalized_mae']['patient']['variable'],result['normalized_mae']['patient']['body_probe'])
        other = predicted.copy(); other['fold'] = 1
        again = assess(pd.concat([predicted,other]),1,repetitions=30)
        self.assertEqual(again['patients'],result['patients'])
        self.assertEqual(again['normalized_mae'],result['normalized_mae'])

    def test_unsupported_component_is_not_silently_dropped(self):
        f,a,v = example(); f['component'] = 'one'
        predicted,_ = crossfit(f,a,v)
        result = assess(predicted,1,repetitions=30)
        self.assertEqual(result['unsupported_occurrences'],120)
        self.assertTrue(all(not c['diagnostic_family_screen'] for c in result['comparisons'].values()))
        self.assertTrue(all(x is None for x in result['decisions'].values()))

    def test_rejects_outer_rows_duplicates_and_nan(self):
        f,a,v = example(); f['role'] = 'test'
        with self.assertRaises(ValueError): crossfit(f,a,v)
        f,a,v = example(); f.loc[0,'patient'] = 'p1'
        with self.assertRaises(ValueError): crossfit(f,a,v)
        f,a,v = example(); f.loc[0,'truth'] = np.nan
        with self.assertRaises(ValueError): crossfit(f,a,v)


if __name__=='__main__': unittest.main()
