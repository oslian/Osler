from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from scripts.train_raw_waveform_event_jepa import _careunit_split, _forward_time_split
from whole_body_jepa.raw_waveform_jepa import (
    MultiHorizonWaveformJEPA,
    MultiTaskResidualWaveformJEPA,
    RawWaveformEventJEPA,
)


class RawWaveformJEPATest(unittest.TestCase):
    def test_external_splits_are_patient_disjoint(self) -> None:
        subjects = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
        careunits = np.asarray(["MICU", "SICU", "MICU", "MICU", "SICU", "SICU", "CCU", "CCU"])
        train, test, _ = _careunit_split(subjects, careunits)
        self.assertTrue(set(subjects[train]).isdisjoint(set(subjects[test])))

        times = pd.Series(pd.date_range("2024-01-01", periods=len(subjects), freq="h"))
        train, test = _forward_time_split(subjects, times)
        self.assertTrue(set(subjects[train]).isdisjoint(set(subjects[test])))

    def test_forward_and_losses(self) -> None:
        model = RawWaveformEventJEPA(20)
        output = model(
            torch.randn(8, 20),
            torch.randn(8, 3, 4, 256),
            torch.ones(8, 3, 4),
            torch.randn(8, 4, 256),
            torch.ones(8, 4),
        )
        self.assertEqual(tuple(output["logit"].shape), (8,))
        self.assertEqual(tuple(output["context_latent"].shape), (8, 64))
        self.assertTrue(torch.isfinite(model.jepa_loss(output)))
        self.assertTrue(
            torch.isfinite(model.event_loss(output, torch.randint(0, 2, (8,)).float()))
        )

    def test_multitask_residual_model(self) -> None:
        model = MultiTaskResidualWaveformJEPA(20)
        output = model(
            torch.randn(8, 20),
            torch.randn(8, 3, 4, 256),
            torch.ones(8, 3, 4),
            torch.randn(8, 4, 256),
            torch.ones(8, 4),
        )
        self.assertTrue(
            torch.allclose(output["logit"], output["table_logit"] + output["residual_logit"])
        )
        loss = model.multitask_loss(
            output,
            torch.randint(0, 2, (8,)).float(),
            torch.randn(8, 4),
        )
        self.assertTrue(torch.isfinite(loss))

    def test_four_scale_morphology_model(self) -> None:
        model = RawWaveformEventJEPA(20, scale_count=4)
        output = model(
            torch.randn(4, 20),
            torch.randn(4, 4, 4, 256),
            torch.ones(4, 4, 4),
            torch.randn(4, 4, 256),
            torch.ones(4, 4),
        )
        self.assertEqual(tuple(output["logit"].shape), (4,))

    def test_multi_horizon_loss(self) -> None:
        model = MultiHorizonWaveformJEPA(20, horizon_count=4)
        output = model(
            torch.randn(8, 20),
            torch.randn(8, 3, 4, 256),
            torch.ones(8, 3, 4),
            torch.randn(8, 4, 256),
            torch.ones(8, 4),
        )
        labels = torch.randint(0, 2, (8, 4)).float()
        mask = torch.ones_like(labels)
        self.assertEqual(tuple(output["logit"].shape), (8, 4))
        self.assertTrue(torch.isfinite(model.multihorizon_loss(output, labels, mask)))


if __name__ == "__main__":
    unittest.main()
