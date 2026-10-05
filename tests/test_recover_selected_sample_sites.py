import unittest
from scripts.recover_selected_sample_sites import unique_match


class MatchTests(unittest.TestCase):
    def test_unique(self):
        self.assertEqual(unique_match([{'site':'a'}]),{'site':'a'})

    def test_none_or_multiple_fail(self):
        for matches in ([],[1,2]):
            with self.assertRaises(ValueError):unique_match(matches)
