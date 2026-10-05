import unittest
from scripts.audit_joint_retention_feasibility import classify


class RetentionFeasibilityTest(unittest.TestCase):
    def test_no_improving_checkpoint(self):
        self.assertEqual(classify([1,2,1.1],2),'never_improved_in_observed_epochs')

    def test_selection_tradeoff(self):
        self.assertEqual(classify([1,.8,1.1],2),'improved_then_not_retained')

    def test_retained(self):
        self.assertEqual(classify([1,.8,.9],2),'selected_not_worse_than_initial')
