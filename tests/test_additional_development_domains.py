import unittest
from scripts.inventory_additional_development_domains import disconnected_candidates


class DevelopmentDomainTests(unittest.TestCase):
    def test_transitive_bridge_cannot_enter(self):
        h={'known':{'a'},'bridge1':{'a','b'},'bridge2':{'b','c'},
           'apparently_new':{'c'},'new':{'d'}}
        people,_=disconnected_candidates(h,{'a'},{'known'})
        self.assertEqual(people,{'new'})

    def test_forbidden_person_excludes_entire_component(self):
        h={'held':{'x','y'},'other':{'y'},'candidate':{'z'}}
        people,_=disconnected_candidates(h,set(),{'held'})
        self.assertEqual(people,{'candidate'})

    def test_order_does_not_change_selection(self):
        h={'p':{'b','a'},'q':{'c'},'r':{'d'}}
        a,_=disconnected_candidates(h,{'a'},set())
        b,_=disconnected_candidates(dict(reversed(list(h.items()))),{'a'},set())
        self.assertEqual(a,b)
