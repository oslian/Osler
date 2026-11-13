import unittest
import pandas as pd
from scripts.audit_source_site_alignment import unique_site_map


class SourceSitesTests(unittest.TestCase):
    def test_never_assign_ambiguous_patient(self):
        frame = pd.DataFrame({'uniquepid': ['a', 'a', 'b'], 'hospitalid': [1, 2, 1], 'role': ['fit'] * 3})
        mapping, ambiguous = unique_site_map(frame)
        self.assertEqual(mapping, {'b': '1'})
        self.assertEqual(ambiguous, {'a'})

    def test_role_overlap_fails(self):
        frame = pd.DataFrame({'uniquepid': ['a', 'a'], 'hospitalid': [1, 1], 'role': ['fit', 'encoder_tune']})
        with self.assertRaises(ValueError):
            unique_site_map(frame)
