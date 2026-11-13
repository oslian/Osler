import unittest
from copy import deepcopy
from scripts.select_reconstruction_bias_polish import keep_correction


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.group={'patients':40,'comparison':dict(sites=4,mean=-.1,site_mean=-.1,patient_p=.0001,site_p=.0001)}

    def test_both_groups_required(self):
        self.assertFalse(keep_correction({'broad':self.group}))
        self.assertTrue(keep_correction({'selected':self.group,'broad':self.group}))

    def test_site_failure_rejected(self):
        other=deepcopy(self.group);other['comparison']['site_p']=.01
        self.assertFalse(keep_correction({'selected':other,'broad':self.group}))
