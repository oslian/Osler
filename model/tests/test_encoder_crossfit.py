import unittest
from scripts.preflight_encoder_crossfit import components, make_folds


class EncoderCrossfitTests(unittest.TestCase):
    def test_patient_bridges_sites_transitively(self):
        sites = {'a': {'1','2'}, 'b': {'2','3'}, 'c': {'3'}, 'd': {'4'}}
        c = components(sites)
        self.assertEqual(c['a'], c['c'])
        self.assertNotEqual(c['a'], c['d'])

    def test_roles_isolate_all_history_sites_and_cover_once(self):
        sites = {str(i): {str(i)} for i in range(12)}
        sites['bridge'] = {'0','1'}
        folds = make_folds(sites)
        self.assertEqual(folds, make_folds(dict(reversed(list(sites.items())))))
        for person in sites:
            self.assertEqual(sum(f[person]=='test' for f in folds), 1)
        for f in folds:
            self.assertEqual(f['0'], f['1'])
            self.assertEqual(f['0'], f['bridge'])

    def test_insufficient_components_fail_closed(self):
        with self.assertRaises(ValueError):
            make_folds({'a': {'1','2'}, 'b': {'2','3'}})
