import unittest
import pandas as pd
from scripts.activate_additional_offsets import decide,source_decisions


class ActivationTests(unittest.TestCase):
    def cv(self,values,complete=True):
        return dict(complete=complete,groups=[dict(status='fitted',delta=x) for x in values])

    def test_fixed_rule(self):
        self.assertTrue(decide(self.cv([-1]*4))['active'])
        self.assertFalse(decide(self.cv([1]*4))['active'])
        self.assertFalse(decide(self.cv([-1]*3))['active'])
        self.assertFalse(decide(self.cv([-1]*4,False))['active'])
        self.assertFalse(decide(self.cv([-1,-1,-1,3]))['active'])

    def test_same_rule_all_arms_and_exact_zero(self):
        f=pd.DataFrame(dict(patient=[str(i) for i in range(120)],component=['a']*30+['b']*30+['c']*30+['d']*30,
                            truth=2.,real=0.,process=2.,population=1.))
        r=source_decisions(f)
        self.assertEqual(r['real']['offset'],2.)
        self.assertTrue(r['real']['active'])
        self.assertEqual(r['population']['offset'],1.)
        self.assertFalse(r['process']['active'])
        self.assertEqual(r['process']['offset'],0.)


if __name__=='__main__':unittest.main()
