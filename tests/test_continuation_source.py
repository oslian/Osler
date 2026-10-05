import unittest
import numpy as np
import pandas as pd

from scripts.diagnose_continuation_source import compare_views, outer_errors, source_harms


class SourceTests(unittest.TestCase):
    def test_source_groups_do_not_hide_harm_in_pooled_mean(self):
        frame=pd.DataFrame(dict(patient=[str(i) for i in range(130)],target='x',
            component=['a']*30+['b']*100,real=[3.]*30+[0.]*100,process=2.,population=2.,truth=0.,fit_scale=1.))
        roles={p:'selection' for p in frame.patient}
        r=source_harms(frame,roles)['x']['population']
        self.assertLess(frame.real.mean(),2.)
        self.assertEqual(r['supported_mean_harm_components'],1)
        roles['0']='test'
        with self.assertRaises(ValueError):source_harms(frame,roles)

    def test_sampling_and_transport_are_separate(self):
        full=np.tile([1.,2.,2.],(5,1));sample=np.tile([3.,2.,2.],(5,1))
        later=np.tile([2.5,2.,2.],(5,1));outer=np.tile([1.5,2.,2.],(5,1))
        r=compare_views(full,sample,later,full,outer)
        self.assertEqual(r['comparisons']['process']['full_source8_monitor8_sign_disagreements'],5)
        self.assertEqual(r['change_sign_disagreements'],5)

    def test_reject_changed_reference_and_missing_folds(self):
        x=np.tile([1.,2.,2.],(5,1));y=x.copy();y[0,2]=3
        with self.assertRaises(ValueError):compare_views(x,x,y,x,x)
        with self.assertRaises(ValueError):compare_views(x[:4],x,x,x,x)

    def test_weighting_retains_small_components(self):
        groups=[dict(fold=f,patients=n,normalized_mae=dict(real=e,process=2.,population=3.))
                for f in range(5) for n,e in ((1,4.),(100,1.))]
        a=dict(targets=dict(x=dict(base_assessment=dict(per_component=groups))))
        self.assertAlmostEqual(outer_errors(a,'x','patient')[0][0],104/101)
        self.assertEqual(outer_errors(a,'x','component')[0][0],2.5)

    def test_unsupported_outer_fold_is_not_a_disagreement(self):
        x=np.tile([1.,2.,2.],(5,1));outer=x.copy();outer[0]=np.nan
        r=compare_views(x,x,x,outer,outer)
        self.assertEqual(r['supported_change_comparisons'],4)
        self.assertEqual(r['change_sign_disagreements'],0)
        self.assertIsNone(r['outer_real_change'][0])


if __name__ == '__main__':unittest.main()
