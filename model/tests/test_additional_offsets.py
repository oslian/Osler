import unittest
import pandas as pd
from scripts.calibrate_additional_reconstruction import fit_offsets, compare, target_union


class OffsetTests(unittest.TestCase):
    def test_sparse_target_absent_first_fold(self):
        frames=[pd.DataFrame(dict(target=['a'])),pd.DataFrame(dict(target=['a','b']))]
        self.assertEqual(target_union(frames),['a','b'])

    def make(self):
        return pd.DataFrame(dict(patient=[str(i) for i in range(360)],
            component=['a']*300+['b']*30+['c']*30,truth=[0.]*300+[10.]*30+[20.]*30,
            real=0.,process=1.,population=2.))

    def test_component_equal_not_patient_median(self):
        r=fit_offsets(self.make())
        self.assertEqual(r['offsets'],dict(real=10.,process=9.,population=8.))
        self.assertEqual(r['status'],'fitted')

    def test_support_and_duplicates(self):
        f=self.make();f=f.loc[f.component!='c']
        self.assertEqual(fit_offsets(f)['offsets'],dict(real=0.,process=0.,population=0.))
        with self.assertRaises(ValueError):fit_offsets(pd.concat([f,f]))

    def test_exact_lower_tie_and_shift(self):
        f=pd.DataFrame(dict(patient=[str(i) for i in range(120)],component=['a']*30+['b']*30+['c']*30+['d']*30,
            truth=[0.]*60+[10.]*60,real=0.,process=0.,population=0.))
        self.assertEqual(fit_offsets(f)['offsets']['real'],0.)
        f.truth+=3
        self.assertEqual(fit_offsets(f)['offsets']['real'],3.)

    def test_unchanged_is_not_improvement(self):
        f=self.make();f['fit_scale']=1.;f['uncorrected_real']=f.real
        r=compare(f,1,100)
        self.assertFalse(r['incremental_descriptive_screen'])
        self.assertEqual(r['comparisons']['uncorrected_real']['patient']['mean'],0.)


if __name__=='__main__':unittest.main()
