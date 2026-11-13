import unittest
from scripts.build_reconstruction_nested_selection import separate_tune


class NestedSelectionTests(unittest.TestCase):
    def test_preserves_existing_roles_and_whole_sites(self):
        roles={'a':'fit','b':'tune','c':'tune','d':'tune','e':'evaluation','f':'reserved'}
        components={'a':0,'b':1,'c':1,'d':2,'e':3,'f':4}
        new=separate_tune(roles,components)
        for p in ('a','e','f'):self.assertEqual(new[p],roles[p])
        self.assertEqual(new['b'],new['c'])
        self.assertNotEqual(new['c'],new['d'])
        self.assertEqual(new,separate_tune(dict(reversed(list(roles.items()))),components))

    def test_rejects_component_overlap(self):
        with self.assertRaises(ValueError):separate_tune({'a':'fit','b':'tune'},{'a':0,'b':0})

    def test_rejects_one_tune_site(self):
        with self.assertRaises(ValueError):separate_tune({'a':'tune','b':'tune'},{'a':0,'b':0})
