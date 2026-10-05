from pathlib import Path
import unittest

from whole_body_jepa import load_release


ROOT = Path(__file__).resolve().parents[1]


class EventRiskTests(unittest.TestCase):
    def test_release_contains_only_fully_audited_event_heads(self) -> None:
        assets = load_release(ROOT / "models")
        expected = {"hypoxemia_event@1h", "hypoxemia_event@3h"}
        self.assertEqual(set(assets.event_heads["validated_heads"]), expected)
        for cell in expected:
            report = assets.event_heads["reports"][cell]
            self.assertTrue(report["runtime_eligible"])
            self.assertEqual(sum(split["pass"] for split in report["patient_splits"]), 7)
            self.assertTrue(report["hospital_holdout"]["pass"])
            self.assertTrue(report["careunit_holdout"]["pass"])
            self.assertTrue(report["forward_time"]["pass"])


if __name__ == "__main__":
    unittest.main()
