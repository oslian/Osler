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

        for module in (site_adaptive_external_audit,):
            self.assertEqual(module.LOCKBOX_DATABASES, {"SICdb", "HiRID"})       # Zigong and Emory were scored: development data
            for database in ("SICdb", "HiRID"):
                with self.assertRaises(ValueError):
                    module.run(argparse.Namespace(database=database, protocol=None))

    def test_lockbox_scoring_needs_the_frozen_invocation_and_an_addendum(self):
        import json
        import tempfile
        from osler_jepa import frozen_protocol
        import site_adaptive_external_audit as audit

        with tempfile.TemporaryDirectory() as folder:
            previous, frozen_protocol.LEDGER = frozen_protocol.LEDGER, Path(folder) / "ledger"
            try:
                import sys
                argv, sys.argv = sys.argv, ["audit", "--database", "SICdb", "--external-template", str(Path(folder) / "x_{horizon}h.parquet"),
                                            "--output", str(Path(folder) / "out.json"), "--protocol", str(Path(folder) / "p.json")]
                try:
                    args = audit.parse_args()
                finally:
                    sys.argv = argv
                protocol = Path(folder) / "p.json"
                protocol.write_text(json.dumps({"frozen_code_sha256": {}, "deployment_form_arguments": {**audit.form_arguments(args), "samples": 3}}))
                with self.assertRaises(RuntimeError):                               # invocation differs from the frozen form
                    audit.lockbox_step(args)
                protocol.write_text(json.dumps({"frozen_code_sha256": {}, "deployment_form_arguments": audit.form_arguments(args)}))
                with self.assertRaises(RuntimeError):                               # no addendum receipt yet
                    audit.lockbox_step(args)
                files = [str(Path(folder) / f"x_{h}h.parquet") for h in (1, 3, 6, 12, 24, 48)]
                for name in files:
                    Path(name).write_bytes(name.encode())
                frozen_protocol.claim_step(protocol, "addendum_SICdb")
                frozen_protocol.record_step(protocol, "addendum_SICdb", files[:1], [])
                with self.assertRaises(RuntimeError):                               # every horizon file must be recorded
                    audit.lockbox_step(args)
                (frozen_protocol.LEDGER / frozen_protocol.sha256(protocol) / "addendum_SICdb.json").unlink()
                frozen_protocol.record_step(protocol, "addendum_SICdb", files, [])
                self.assertEqual(audit.lockbox_step(args), "continuous_SICdb")
                with self.assertRaises(FileExistsError):                            # scored once
                    audit.lockbox_step(args)
            finally:
                frozen_protocol.LEDGER = previous


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
