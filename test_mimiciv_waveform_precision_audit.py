import unittest

from mimiciv_waveform_precision_audit import is_waveform_feature, wave_feature_sets
import pandas as pd


class MimicivWaveformPrecisionAuditTests(unittest.TestCase):
    def test_feature_registry_accepts_legacy_and_beat_prefixes(self):
        self.assertTrue(is_waveform_feature("wave_ecg_1_median"))
        self.assertTrue(is_waveform_feature("wavebeat_ecg_1_rate_per_min"))
        self.assertFalse(is_waveform_feature("heart_rate_t"))

    def test_consensus_profile_removes_redundant_leads(self):
        frame = pd.DataFrame({
            "heart_rate_t": [80.0],
            "heart_rate_tp1": [82.0],
            "wavebeat_ecg_1_rate_per_min": [80.0],
            "wavebeat_ecg_consensus_rate_per_min": [80.0],
            "wavebeat_ecg_consensus_iqr": [2.0],
            "wavebeat_ecg_consensus_agreement_fraction": [1.0],
        })
        profile = wave_feature_sets(frame, "heart_rate", profile="consensus")
        self.assertNotIn("wavebeat_ecg_1_rate_per_min", profile["wave"])
        self.assertIn("wavebeat_ecg_consensus_rate_per_min", profile["wave"])


if __name__ == "__main__":
    unittest.main()
