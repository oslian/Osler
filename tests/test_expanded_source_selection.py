import unittest
from scripts.replay_expanded_source_selection import summarize


class SelectionReplayTest(unittest.TestCase):
    def test_scopes_and_target_equal_macro(self):
        rows=[dict(target=t,scope=s,truth=0.,prediction=v) for t,s,v in
              [('a','original',1.),('a','additional',3.),('a','additional',5.),
               ('b','original',2.),('b','additional',4.)]]
        r=summarize(rows,['a','b'])
        self.assertEqual(r['original']['macro'],1.5)
        self.assertEqual(r['additional']['macro'],4.)
        self.assertEqual(r['all']['macro'],3.)

    def test_missing_support_rejected(self):
        with self.assertRaises(ValueError):
            summarize([],['a'])
