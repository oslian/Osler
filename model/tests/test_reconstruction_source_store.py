from pathlib import Path
import tempfile
import unittest

import torch

from scripts.reconstruction_source_store import SourceStore, source_shards
from whole_body_jepa.multisource_intervals import sha256


class SourceStoreTest(unittest.TestCase):
    def test_lazy_values_and_bounded_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            shards = []
            for patient in range(3):
                path = Path(directory) / f"{patient}.pt"
                torch.save(dict(patient=patient, role="fit", samples={0: {
                    "truth_normalized": float(patient), "tokens": torch.ones(2, 3)}}), path)
                shards.append(dict(patient=patient, role="fit", path=str(path), sha256=sha256(path)))
            store = SourceStore(shards, [0], cache_size=1)
            self.assertEqual(len(store.groups[0]), 3)
            for patient in range(3):
                self.assertEqual(store[patient, 0]["truth_normalized"], patient)
            self.assertEqual(store._cached_read.cache_info().currsize, 1)
            path = Path(shards[0]["path"])
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                store[0, 0]

    def test_expansion_keeps_selection_fixed(self):
        joint = dict(variables=["x"], shards=[
            dict(patient=1, group="selected", role="fit"),
            dict(patient=2, group="broad", role="fit"),
            dict(patient=3, group="broad", role="encoder_tune")])
        expanded = dict(variables=["x"], shards=[
            dict(patient=2, role="fit"), dict(patient=4, role="fit"),
            dict(patient=5, role="encoder_tune")])
        result = source_shards(joint, expanded, "expanded", {1: "fit"})
        self.assertEqual({s["patient"] for s in result}, {1, 2, 3, 4})
        self.assertEqual([s for s in result if s["role"] == "encoder_tune"],
                         [joint["shards"][2]])
        expanded["shards"].append(dict(patient=3, role="fit"))
        with self.assertRaisesRegex(ValueError, "crosses"):
            source_shards(joint, expanded, "expanded", {1: "fit"})

    def test_nested_holdout_rejected(self):
        joint = dict(variables=["x"], shards=[dict(patient=1, group="selected", role="fit")])
        expanded = dict(variables=["x"], shards=[])
        with self.assertRaisesRegex(ValueError, "Nested"):
            source_shards(joint, expanded, "small", {1: "evaluation"})

    def test_rejects_wrong_patient(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pt"
            torch.save(dict(patient=2, role="fit", samples={}), path)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                SourceStore([dict(patient=1, role="fit", path=str(path),
                                  sha256=sha256(path))], [0])
