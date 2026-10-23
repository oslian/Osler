import unittest
from pathlib import Path

import pandas as pd

from multi_sample_external_audit import mimic_sample, summarize
from mimic_to_eicu_49_cell_external_audit import cap_by_subject, stable_seed


class MultiSampleTests(unittest.TestCase):
    def test_sample_zero_reproduces_the_original_audit_sample(self):
        frame = pd.DataFrame({"subject_id": [f"MIMIC:{i // 3}" for i in range(300)], "x": range(300)})
        original = cap_by_subject(frame, 90, stable_seed("MIMIC-IV", "glucose", 6)).reset_index(drop=True)
        pd.testing.assert_frame_equal(mimic_sample(frame, target="glucose", horizon=6, sample=0, rows=90), original)
        other = mimic_sample(frame, target="glucose", horizon=6, sample=1, rows=90)
        self.assertNotEqual(set(other["subject_id"]), set(original["subject_id"]))

    def test_robust_needs_eighty_percent_full_passes(self):
        results = [{"full": f, "point": True, "interval": f, "coverage": 0.9} for f in (True, True, True, True, False)]
        self.assertTrue(summarize(results)["robust"])
        results[3]["full"] = False
        self.assertFalse(summarize(results)["robust"])


class LockboxGuardTests(unittest.TestCase):
    def test_lockbox_databases_refuse_to_run_without_a_protocol(self):
        import argparse
        import site_adaptive_external_audit
        import multi_sample_external_audit

        for module in (site_adaptive_external_audit, multi_sample_external_audit):
            for database in ("SICdb", "HiRID", "Emory"):
                with self.assertRaises(ValueError):
                    module.run(argparse.Namespace(database=database, protocol=None))


    def test_protocol_verification_covers_code_and_training_data(self):
        import json
        import os
        import tempfile
        from external_value_conditional_interval_audit import sha256, verify_protocol

        with tempfile.TemporaryDirectory() as folder:
            code, data = Path(folder) / "code.py", Path(folder) / "data.parquet"
            code.write_text("x = 1")
            data.write_bytes(b"rows")
            protocol = Path(folder) / "protocol.json"
            protocol.write_text(json.dumps({"frozen_code_sha256": {str(code): sha256(code)},
                                            "frozen_training_data_sha256": {str(data): sha256(data)}}))
            verify_protocol(protocol)
            data.write_bytes(b"changed rows")
            with self.assertRaises(RuntimeError):
                verify_protocol(protocol)

if __name__ == "__main__":
    unittest.main()
