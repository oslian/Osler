from pathlib import Path
import json
import unittest

import numpy as np
import pandas as pd

from scripts.build_cross_database_registry import build_registry
from whole_body_jepa.replication import validate_scored_lockbox


ROOT = Path(__file__).resolve().parents[1]


class ReplicationTests(unittest.TestCase):
    def test_release_has_frozen_cross_database_inventory(self) -> None:
        registry = build_registry(ROOT / "models")
        self.assertEqual(registry["strict_replicated_cells"], 27)
        self.assertEqual(registry["challenge_cells"], 14)
        self.assertFalse(registry["mimic_labels_used_for_selection"])
        self.assertEqual(registry["transfer_type"], "frozen_weight_direct_transfer")

    def test_checked_in_registry_matches_release(self) -> None:
        checked_in = json.loads(
            (ROOT / "models/cross_database_replication_registry.json").read_text()
        )
        generated = build_registry(ROOT / "models")
        self.assertEqual(checked_in, generated)

    def test_lockbox_gate_requires_patient_domain_and_time(self) -> None:
        rows = []
        for patient in range(20):
            for repeat in range(4):
                truth = float(patient + repeat)
                rows.append(
                    {
                        "patient_id": patient,
                        "domain_id": patient % 4,
                        "time_block": repeat % 2,
                        "cell": "map@1h",
                        "current": truth - 2.0,
                        "truth": truth,
                        "candidate": truth + 0.1,
                        "persistence": truth - 2.0,
                    }
                )
        report = validate_scored_lockbox(
            pd.DataFrame(rows), bootstrap_repetitions=300
        )
        cell = report["cells"][0]
        self.assertTrue(cell["strict_frozen_replication_pass"])
        self.assertEqual(report["strict_pass_cells"], ["map@1h"])
        self.assertGreater(cell["direction_accuracy"], 0.9)

    def test_missing_domain_fails_closed(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": np.arange(20),
                "cell": "map@1h",
                "truth": np.arange(20, dtype=float),
                "candidate": np.arange(20, dtype=float),
                "persistence": np.arange(20, dtype=float) + 1.0,
            }
        )
        report = validate_scored_lockbox(frame, bootstrap_repetitions=100)
        self.assertFalse(report["cells"][0]["strict_frozen_replication_pass"])
        self.assertIn("domain_gate", report["cells"][0]["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
