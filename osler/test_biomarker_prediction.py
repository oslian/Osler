import unittest

from osler_jepa.biomarker_prediction import (
    MIMIC_BIOMARKER_ITEMIDS,
    prediction_contract,
)


class BiomarkerPredictionPolicyTests(unittest.TestCase):
    def test_next_measurement_time_is_disabled_for_every_tier(self):
        for target in ("hba1c", "cortisol", "troponin_i", "germline_genetics"):
            self.assertFalse(prediction_contract(target)["next_measurement_time_allowed"])

    def test_routine_target_allows_value_candidate_without_promotion(self):
        contract = prediction_contract("HbA1c")
        self.assertEqual(contract["tier"], "routine_value")
        self.assertTrue(contract["numeric_future_candidate"])
        self.assertEqual(contract["status"], "candidate_only")

    def test_specialty_marker_uses_risk_not_numeric_future(self):
        contract = prediction_contract("CA 19-9")
        self.assertEqual(contract["tier"], "specialty_risk")
        self.assertFalse(contract["numeric_future_candidate"])
        self.assertTrue(contract["future_abnormal_risk_candidate"])

    def test_genetics_is_observed_only(self):
        contract = prediction_contract("germline genetics")
        self.assertEqual(contract["status"], "observed_only")
        self.assertFalse(contract["current_estimate_allowed"])

    def test_itemids_cover_requested_new_numeric_targets(self):
        expected = {
            "total_cholesterol", "hdl_cholesterol", "ldl_cholesterol", "hba1c",
            "iron", "vitamin_d_25oh", "vitamin_b12", "estradiol", "progesterone",
            "testosterone_total", "afp", "ca_125", "ca_19_9", "psa",
            "rheumatoid_factor",
        }
        self.assertTrue(expected.issubset(MIMIC_BIOMARKER_ITEMIDS))


if __name__ == "__main__":
    unittest.main()
