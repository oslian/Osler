import unittest

from whole_body_joint_gate import (
    JOINT_MOVEMENT_REQUIREMENTS,
    apply_joint_movement_policy,
)


class JointMovementPolicyTests(unittest.TestCase):
    @staticmethod
    def _passing_entry():
        return {name: True for name in JOINT_MOVEMENT_REQUIREMENTS}

    def test_non_cross_target_can_move_without_unrun_attribution(self):
        entry = self._passing_entry()
        result = apply_joint_movement_policy(
            entry,
            cross_system_evaluated=False,
        )
        self.assertTrue(result["validated_for_joint_runtime"])
        self.assertFalse(result["cross_organ_claim_allowed"])

    def test_cross_organ_claim_requires_every_external_split(self):
        entry = self._passing_entry()
        entry.update(
            {
                "patient_cross_system": True,
                "hospital_cross_system": True,
                "care_unit_cross_system": True,
                "time_cross_system": False,
            }
        )
        result = apply_joint_movement_policy(
            entry,
            cross_system_evaluated=True,
        )
        self.assertTrue(result["validated_for_joint_runtime"])
        self.assertFalse(result["cross_organ_claim_allowed"])

        entry["time_cross_system"] = True
        result = apply_joint_movement_policy(
            entry,
            cross_system_evaluated=True,
        )
        self.assertTrue(result["cross_organ_claim_allowed"])

    def test_accuracy_failure_still_fails_closed(self):
        entry = self._passing_entry()
        entry["patient_validated_router_bootstrap"] = False
        result = apply_joint_movement_policy(
            entry,
            cross_system_evaluated=False,
        )
        self.assertFalse(result["validated_for_joint_runtime"])


if __name__ == "__main__":
    unittest.main()
