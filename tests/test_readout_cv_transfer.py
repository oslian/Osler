import unittest
from scripts.audit_readout_cv_transfer import comparison


class TransferTests(unittest.TestCase):
    def test_equal_groups_not_patient_counts(self):
        choice={'name':'1.0','cv_scores':{'unchanged':2.,'1.0':1.5}}
        groups={'selected':{'patients':1,'comparison':{'site_mean':.2}},'broad':{'patients':1000,'comparison':{'site_mean':-.1}}}
        row=comparison(choice,groups)
        self.assertAlmostEqual(row['held_site_delta'],.05)
        self.assertAlmostEqual(row['optimism'],.55)
