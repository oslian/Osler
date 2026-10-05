import unittest
from scripts.audit_bun_offset_transfer import offsets


class OffsetTests(unittest.TestCase):
    def test_signed_median(self):
        rows=[dict(truth=t,predictions={'full':t+2,'no_body':t-1}) for t in (1,2,9)]
        self.assertEqual(offsets(rows),{'full':-2.,'no_body':1.})

    def test_rejects_invalid(self):
        for rows in ([],[dict(truth=float('nan'),predictions={'full':0,'no_body':0})]):
            with self.assertRaises(ValueError): offsets(rows)
