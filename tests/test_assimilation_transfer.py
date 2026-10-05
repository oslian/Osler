import unittest

import pandas as pd

from scripts.probe_assimilation_transfer import ARMS, choose_keys, seen_people, summarize


class AssimilationTransferTest(unittest.TestCase):
    def test_exposure_indices_map_to_people(self):
        self.assertEqual(seen_people({(0, 1), (1, 0)}, ['a', 'b'], ['x', 'y'],
                                    {'a': 'fit', 'b': 'fit'}), {'a', 'b'})
        for bad in ({('a', 0)}, {(2, 0)}, {(True, 0)}, {(0, -1)}):
            with self.assertRaises(ValueError):
                seen_people(bad, ['a'], ['x'], {'a': 'fit'})
        with self.assertRaises(ValueError):
            seen_people({(0, 0)}, ['a'], ['x'], {'a': 'selection'})

    def test_exclude_whole_adapted_patient_and_unseen_source(self):
        keys = [('a', 0), ('a', 1), ('b', 0), ('c', 0)]
        args = ({'a'}, {'a': 's', 'b': 's', 'c': 't'}, {'s'},
                {'a': 'fit', 'b': 'fit', 'c': 'fit'})
        self.assertEqual(choose_keys(keys, *args), [('b', 0)])
        self.assertEqual(choose_keys(keys[::-1], *args), [('b', 0)])
        with self.assertRaises(ValueError): choose_keys(keys + [keys[0]], *args)

    def test_missing_provenance_and_nonfit_role_fail(self):
        with self.assertRaises(ValueError):
            choose_keys([('a', 0)], set(), {}, {'s'}, {'a': 'fit'})
        with self.assertRaises(ValueError):
            choose_keys([('a', 0)], set(), {'a': 's'}, {'s'}, {'a': 'selection'})

    def test_patient_equal_not_occurrence_equal(self):
        rows = [dict(patient=p, component='c', variable='v', fold=f, truth=0.,
                     epoch10_exposed=True, **{a: value for a in ARMS})
                for p, f, value in [('a', 0, 0.), ('a', 1, 0.), ('b', 0, 3.)]]
        r = summarize(pd.DataFrame(rows), ['v', 'w'])
        self.assertEqual(r['v']['normalized_mae']['patient']['immediate'], 1.5)
        self.assertEqual(r['w']['status'], 'unsupported')
        with self.assertRaises(ValueError): summarize(pd.DataFrame(rows + [rows[0]]), ['v'])


if __name__ == '__main__':
    unittest.main()
