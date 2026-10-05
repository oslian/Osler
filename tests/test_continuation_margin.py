import copy
import unittest

from scripts.diagnose_continuation_margin import compare_components


def rows(real, process):
    return [dict(component=str(i),patients=n,fold=i,
                 raw_mae=dict(real=r,process=p,population=2.),
                 normalized_mae=dict(real=r,process=p,population=2.),
                 real_minus_process=r-p,real_minus_population=r-2.)
            for i,(r,p,n) in enumerate(zip(real,process,(1,100,40)))]


class MarginTests(unittest.TestCase):
    def test_weaker_control_not_model_improvement(self):
        result=compare_components(rows([3]*3,[2]*3),rows([3]*3,[4]*3),7)
        self.assertEqual(result['means']['real_change'],0)
        self.assertLess(result['means']['new_margin'],0)
        self.assertGreater(result['means']['new_real_old_process'],0)
        self.assertEqual(result['harm']['population']['supported_persistent_mean_harm'],2)

    def test_recovery_and_new_harm_retain_small_groups(self):
        result=compare_components(rows([3,1,3],[2]*3),rows([1,3,3],[2]*3),7)
        self.assertEqual(len(result['groups']),3)
        self.assertEqual(result['harm']['population']['recovered'],1)
        self.assertEqual(result['harm']['population']['newly_harmed'],1)
        self.assertEqual(result['harm']['population']['persistent_mean_harm'],1)

    def test_reject_changed_support(self):
        old=rows([1]*3,[2]*3)
        for field,value in [('patients',99),('fold',4),('component','other')]:
            new=copy.deepcopy(old);new[0][field]=value
            with self.assertRaises(ValueError):compare_components(old,new,7)

    def test_reject_changed_population_or_nonfinite(self):
        old=rows([1]*3,[2]*3)
        for field,value in [('population',3.),('real',float('nan'))]:
            new=copy.deepcopy(old);new[0]['raw_mae'][field]=value
            with self.assertRaises(ValueError):compare_components(old,new,7)

    def test_reject_duplicate_components(self):
        old=rows([1]*3,[2]*3)
        with self.assertRaises(ValueError):compare_components(old,[old[0]]*3,7)


if __name__ == '__main__':unittest.main()
