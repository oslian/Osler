import unittest

import pandas as pd

from target_specific_domain_conformal_audit import _split_specs


class TargetSpecificDomainConformalAuditTests(unittest.TestCase):
    def test_single_center_uses_only_available_external_domains(self):
        frame = pd.DataFrame({
            "subject_id": [f"p{index // 2}" for index in range(20)],
            "stay_id": [index // 2 for index in range(20)],
            "hospitalid": ["MIMIC-IV"] * 20,
            "careunit": ["MICU", "SICU"] * 10,
            "anchor_time": pd.date_range("2020-01-01", periods=20, freq="h"),
            "hours_since_onset": [float(index % 2) for index in range(20)],
        })

        names = [name for name, *_ in _split_specs(frame, (0, 1))]

        self.assertIn("patient_seed_0", names)
        self.assertIn("careunit", names)
        self.assertIn("forward_time", names)
        self.assertNotIn("hospital", names)


if __name__ == "__main__":
    unittest.main()
