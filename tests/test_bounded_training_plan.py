import unittest
from collections import Counter
import numpy as np
from scripts.train_bounded_site_source import replacement_plan


class ReplacementPlanTests(unittest.TestCase):
    def test_matched_schedule_and_exact_budget(self):
        groups={g:{c:[(f'{g}{i}',c) for i in range(4)] for c in (1,2)} for g in ('a','b')}
        uniform={g:{c:np.ones(4) for c in (1,2)} for g in groups}
        bounded={g:{c:np.array([.5,.5,1.,2.]) for c in (1,2)} for g in groups}
        a=replacement_plan(groups,uniform,np.random.default_rng(19),draws=10)
        b=replacement_plan(groups,bounded,np.random.default_rng(19),draws=10)
        self.assertEqual([(g,k[1]) for g,k in a],[(g,k[1]) for g,k in b])
        self.assertEqual(Counter((g,k[1]) for g,k in a),Counter({(g,c):10 for g in groups for c in (1,2)}))
        self.assertEqual(a,replacement_plan(groups,uniform,np.random.default_rng(19),draws=10))

    def test_rng_resume(self):
        groups={'a':{0:[('p',0),('q',0)]}}
        weights={'a':{0:np.ones(2)}}
        rng=np.random.default_rng(19)
        replacement_plan(groups,weights,rng,draws=3)
        other=np.random.default_rng();other.bit_generator.state=rng.bit_generator.state
        self.assertEqual(replacement_plan(groups,weights,rng,draws=3),replacement_plan(groups,weights,other,draws=3))
