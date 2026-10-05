import unittest
from scripts.audit_reconstruction_readout_selection import select_source


class SelectionTests(unittest.TestCase):
    def test_tie_keeps_original(self):
        self.assertEqual(select_source({'original':1.,'population':1.,'ridge':1.}),'original')

    def test_selects_lowest(self):
        self.assertEqual(select_source({'original':1.,'population':.9,'ridge':.8}),'ridge')

    def test_invalid(self):
        for scores in ({},{'original':float('nan')}):
            with self.assertRaises(ValueError):select_source(scores)
