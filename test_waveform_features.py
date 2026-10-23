import math
import unittest

import numpy as np

from osler_jepa.waveform_features import (
    aggregate_feature_summary,
    peak_interval_features,
    robust_univariate_features,
    waveform_array_features,
    WaveformFeatureRecord,
)
from osler_jepa.waveform_encoder import beat_level_array_features


class WaveformFeatureTests(unittest.TestCase):
    def test_robust_univariate_features_ignore_nan(self):
        features = robust_univariate_features(np.array([1.0, 2.0, np.nan, 4.0]), "x")

        self.assertAlmostEqual(features["x_valid_fraction"], 0.75)
        self.assertAlmostEqual(features["x_median"], 2.0)
        self.assertGreater(features["x_iqr"], 0)

    def test_empty_channel_returns_nan_features(self):
        features = robust_univariate_features(np.array([np.nan, np.nan]), "x")

        self.assertEqual(features["x_valid_fraction"], 0.0)
        self.assertTrue(math.isnan(features["x_median"]))

    def test_waveform_array_features_group_core_signals(self):
        values = np.array(
            [
                [0.1, 70.0, 0.5, 0.2],
                [0.2, 80.0, 0.7, 0.3],
                [0.3, 60.0, 0.4, 0.4],
            ]
        )
        features = waveform_array_features(values, ["II", "ABP", "Pleth", "Resp"])

        self.assertIn("wave_ecg_1_median", features)
        self.assertIn("wave_arterial_pressure_1_median", features)
        self.assertIn("wave_arterial_pressure_1_hypotension_fraction_lt65", features)
        self.assertIn("wave_pleth_1_iqr", features)
        self.assertIn("wave_respiration_1_std", features)

    def test_peak_interval_features_estimate_rate(self):
        fs = 50.0
        t = np.arange(0, 10, 1 / fs)
        signal = np.sin(2 * np.pi * 1.0 * t)

        features = peak_interval_features(
            signal,
            fs,
            "ecg",
            min_interval_seconds=0.3,
            max_interval_seconds=2.0,
        )

        self.assertGreater(features["ecg_peak_count"], 5)
        self.assertAlmostEqual(features["ecg_peak_rate_per_min"], 60.0, delta=5.0)

    def test_waveform_array_features_include_peak_rates_when_fs_available(self):
        fs = 50.0
        t = np.arange(0, 10, 1 / fs)
        values = np.column_stack([
            np.sin(2 * np.pi * 1.2 * t),
            np.sin(2 * np.pi * 0.25 * t),
        ])

        features = waveform_array_features(values, ["II", "Resp"], sampling_frequency_hz=fs)

        self.assertIn("wave_ecg_1_peak_rate_per_min", features)
        self.assertIn("wave_respiration_1_peak_rate_per_min", features)

    def test_aggregate_summary_has_no_prediction_authority(self):
        record = WaveformFeatureRecord(
            record_name="r1",
            source_path="/tmp/r1",
            sampling_frequency_hz=125.0,
            seconds_read=60.0,
            signals=("II", "ABP"),
            signal_groups=("arterial_pressure", "ecg"),
            features={"wave_ecg_1_median": 0.1},
        )

        summary = aggregate_feature_summary([record])

        self.assertEqual(summary["record_count"], 1)
        self.assertEqual(summary["signal_group_counts"]["ecg"], 1)
        self.assertFalse(summary["boundary"]["prediction_authority"])
        self.assertFalse(summary["boundary"]["raw_samples_committed"])

    def test_beat_encoder_recovers_ecg_and_resp_rates(self):
        fs = 100.0
        t = np.arange(0, 30, 1 / fs)
        beat_times = np.arange(0.5, 30.0, 1 / 1.2)
        ecg = sum(np.exp(-((t - beat) / 0.025) ** 2) for beat in beat_times)
        values = np.column_stack([
            ecg,
            np.sin(2 * np.pi * 0.25 * t),
        ])

        features = beat_level_array_features(values, ["II", "Resp"], fs)

        self.assertAlmostEqual(features["wavebeat_ecg_1_rate_per_min"], 72.0, delta=8.0)
        self.assertAlmostEqual(features["wavebeat_respiration_1_rate_per_min"], 15.0, delta=3.0)
        self.assertGreater(features["wavebeat_ecg_1_quality_valid_fraction"], 0.99)
        self.assertIn("wavebeat_ecg_consensus_rate_per_min", features)
        self.assertGreater(features["wavebeat_ecg_consensus_agreement_fraction"], 0.9)

    def test_beat_encoder_emits_arterial_pressure_cycle_features(self):
        fs = 100.0
        t = np.arange(0, 20, 1 / fs)
        pulse = 80.0 + 40.0 * np.maximum(np.sin(2 * np.pi * 1.0 * t), 0.0)

        features = beat_level_array_features(pulse[:, None], ["ABP"], fs)

        self.assertIn("wavebeat_arterial_pressure_1_sbp_median", features)
        self.assertIn("wavebeat_arterial_pressure_1_dbp_median", features)
        self.assertGreater(features["wavebeat_arterial_pressure_1_beat_count"], 10)


if __name__ == "__main__":
    unittest.main()
