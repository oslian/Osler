import unittest
import pandas as pd
from whole_body_jepa.site_partitions import site_partition


class SitePartitionTests(unittest.TestCase):
    def test_transitive_history_reservation(self):
        q = pd.DataFrame({'uniquepid': ['a','a','b','b','c','d','e','f','g'],
                          'hospitalid': [1,2,2,3,4,5,6,7,8]})
        roles, _ = site_partition(q, {'a'})
        self.assertEqual(roles['b'], 'reserved')
        mapped = q.assign(role=q.uniquepid.map(roles))
        self.assertTrue((mapped.groupby('hospitalid').role.nunique()==1).all())
        self.assertEqual(set(roles.values()), {'fit','tune','evaluation','reserved'})
        self.assertEqual(roles, site_partition(q.iloc[::-1], {'a'})[0])

    def test_missing_and_unknown_rejected(self):
        q = pd.DataFrame({'uniquepid': ['a','b','c','d'], 'hospitalid': [1,2,3,4]})
        with self.assertRaises(ValueError):
            site_partition(q, {'unknown'})
        q.loc[0,'hospitalid'] = None
        with self.assertRaises(ValueError):
            site_partition(q, set())

    def test_insufficient_components(self):
        q = pd.DataFrame({'uniquepid': ['a','b'], 'hospitalid': [1,2]})
        with self.assertRaises(ValueError):
            site_partition(q, set())
