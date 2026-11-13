import unittest
from scripts.audit_confirmed_reconstruction_heads import evidence, confirmed


class ConfirmationTests(unittest.TestCase):
    def test_equal_predictions_fail(self):
        row = evidence([0.] * 30, [i % 3 for i in range(30)])
        self.assertFalse(confirmed({'original': row, 'population': row}, 30))

    def test_support_and_multiplicity(self):
        row = dict(sites=3, mean=-1., site_mean=-1., patient_p=.0001, site_p=.0001)
        self.assertTrue(confirmed({'original': row, 'population': row}, 30))
        self.assertFalse(confirmed({'original': row}, 29))
        self.assertFalse(confirmed({'original': dict(row, site_p=.01)}, 30))

    def test_conflicting_sites_fail(self):
        row = dict(sites=3, mean=-1., site_mean=.1, patient_p=.0001, site_p=.0001)
        self.assertFalse(confirmed({'original': row}, 30))
