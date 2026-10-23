import unittest

import pandas as pd

from mimiciv_cross_database_coverage_audit import (
    physiology_forecast_feature_columns,
)


class MimicIVCrossDatabaseCoverageAuditTests(unittest.TestCase):
    def test_no_treatment_audit_excludes_all_treatment_columns(self):
        frame = pd.DataFrame({
            "subject_id": [1, 2],
            "stay_id": [10, 20],
            "creatinine_t": [1.0, 1.2],
            "creatinine_age_hr": [0.5, 0.7],
            "creatinine_tp24": [1.1, 1.3],
            "map_t": [70.0, 75.0],
            "hist_fluids": [1, 0],
            "act_fluids": [0, 1],
        })

        features = physiology_forecast_feature_columns(frame, "creatinine")

        self.assertIn("map_t", features)
        self.assertNotIn("hist_fluids", features)
        self.assertNotIn("act_fluids", features)


if __name__ == "__main__":
    unittest.main()
