from __future__ import annotations

import unittest

import numpy as np
import torch

from whole_body_jepa.waveform_event import (
    cough_effort_score,
    cough_reflex_score,
    incident_sustained_hypoxemia,
    oxygen_support_level,
    productive_cough,
    signed_leading_score,
    waveform_event_decision,
)
from whole_body_jepa.waveform_jepa import WaveformEventJEPA


class WaveformEventTest(unittest.TestCase):
    def test_dense_sustained_hypoxemia_is_not_missed(self) -> None:
        times = np.arange(0.0, 1801.0, 1.0)
        spo2 = np.full(len(times), 98.0)
        spo2[(times >= 930.0) & (times <= 990.0)] = 90.0
        label, _ = incident_sustained_hypoxemia(times, spo2, 900.0)
        self.assertEqual(label, 1)

    def test_bedside_assessment_parsers(self) -> None:
        self.assertEqual(signed_leading_score("-3 Moderate sedation"), -3.0)
        self.assertEqual(signed_leading_score("+2 Frequent movement"), 2.0)
        self.assertEqual(signed_leading_score("Mild - 2"), 2.0)
        self.assertEqual(cough_reflex_score("Impaired"), 1.0)
        self.assertEqual(cough_effort_score("Strong"), 2.0)
        self.assertEqual(productive_cough("Productive/Congested"), 1.0)
        self.assertEqual(productive_cough("Non-productive"), 0.0)

    def test_sustained_event_rejects_isolated_low_reading(self) -> None:
        times = np.asarray([-300, 0, 60, 120, 180], dtype=float)
        values = np.asarray([97, 96, 91, 96, 97], dtype=float)
        label, _ = incident_sustained_hypoxemia(times, values, 0.0)
        self.assertEqual(label, 0)

    def test_sustained_event_requires_two_separated_low_readings(self) -> None:
        times = np.asarray([-300, 0, 60, 120, 180], dtype=float)
        values = np.asarray([97, 96, 91, 90, 97], dtype=float)
        label, descriptor = incident_sustained_hypoxemia(times, values, 0.0)
        self.assertEqual(label, 1)
        self.assertEqual(descriptor["future_spo2_min"], 90.0)

    def test_current_hypoxemia_is_not_an_incident_event_anchor(self) -> None:
        times = np.asarray([-60, 0, 60, 120], dtype=float)
        values = np.asarray([91, 96, 91, 90], dtype=float)
        label, _ = incident_sustained_hypoxemia(times, values, 0.0)
        self.assertIsNone(label)

    def test_oxygen_support_is_ordered(self) -> None:
        self.assertLess(oxygen_support_level("Room Air"), oxygen_support_level("Nasal cannula"))
        self.assertLess(oxygen_support_level("High flow nasal cannula"), oxygen_support_level("Ventilator"))

    def test_waveform_jepa_shapes_and_loss(self) -> None:
        model = WaveformEventJEPA(12, 20, 5)
        output = model(torch.randn(8, 12), torch.randn(8, 20), torch.randn(8, 5))
        self.assertEqual(tuple(output["logit"].shape), (8,))
        self.assertEqual(tuple(output["predicted_future_latent"].shape), (8, 24))
        loss = model.loss(output, torch.randint(0, 2, (8,)).float())
        self.assertTrue(torch.isfinite(loss))

    def test_candidate_abstains_without_external_lockbox(self) -> None:
        decision = waveform_event_decision(
            0.72,
            ensemble_standard_deviation=0.02,
            quality={
                "wf_300s_pleth_valid_fraction": 1.0,
                "wf_300s_pleth_flatline_fraction": 0.01,
                "wf_300s_ecg_valid_fraction": 1.0,
            },
            external_waveform_lockbox_validated=False,
            calibration_validated=True,
        )
        self.assertTrue(decision["abstained"])
        self.assertIsNone(decision["event_risk_probability"])
        self.assertIn(
            "external_waveform_lockbox_not_validated",
            decision["abstention_reasons"],
        )

    def test_validated_low_disagreement_event_can_report_probability(self) -> None:
        decision = waveform_event_decision(
            0.72,
            ensemble_standard_deviation=0.02,
            quality={
                "wf_300s_pleth_valid_fraction": 1.0,
                "wf_300s_pleth_flatline_fraction": 0.01,
                "wf_300s_respiration_valid_fraction": 1.0,
            },
            external_waveform_lockbox_validated=True,
            calibration_validated=True,
        )
        self.assertFalse(decision["abstained"])
        self.assertEqual(decision["event_risk_probability"], 0.72)


if __name__ == "__main__":
    unittest.main()
