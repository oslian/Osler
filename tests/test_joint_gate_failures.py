import unittest
from scripts.audit_joint_gate_failures import reasons


class GateReasonsTest(unittest.TestCase):
    def test_favorable_mean_is_not_evidence(self):
        self.assertEqual(reasons(dict(mean=-1,site_mean=-1,sites=7,patient_p=.0001,site_p=.01)),['site_evidence_insufficient'])

    def test_sign_failure_not_hidden(self):
        r=reasons(dict(mean=1,site_mean=-1,sites=7,patient_p=.9,site_p=.0001))
        self.assertIn('pooled_mean_not_better',r)

    def test_pass(self):
        self.assertEqual(reasons(dict(mean=-1,site_mean=-1,sites=7,patient_p=.0001,site_p=.0001)),[])
