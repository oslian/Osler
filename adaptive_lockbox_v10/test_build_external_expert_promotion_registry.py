import json
import tempfile
import unittest
from pathlib import Path

from build_external_expert_promotion_registry import METHOD, lockbox_cells, outcome, status
from osler_jepa import frozen_protocol


def cell(robust=True, evaluable=True):
    if not evaluable:
        return {"cell": "x@1h", "evaluable": False, "reason": "insufficient_external_support"}
    return {"cell": "x@1h", "evaluable": True, "methods": {METHOD: {"robust": robust, "full_pass_share": 1.0}}}


class ReplicationStatusTests(unittest.TestCase):
    def test_robust_on_both_lockboxes_is_replicated(self):
        self.assertEqual(status([outcome(cell()), outcome(cell())]), "independently_replicated")

    def test_robust_on_one_is_a_candidate(self):
        self.assertEqual(status([outcome(cell()), outcome(cell(robust=False))]), "candidate")
        self.assertEqual(status([outcome(cell()), outcome(cell(evaluable=False))]), "candidate")
        self.assertEqual(status([outcome(None), outcome(cell())]), "candidate")

    def test_robust_on_neither_is_not_promoted(self):
        self.assertEqual(status([outcome(cell(robust=False)), outcome(cell(evaluable=False))]), "not_promoted")

    def test_only_the_recorded_scored_output_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            previous, frozen_protocol.LEDGER = frozen_protocol.LEDGER, Path(folder) / "ledger"
            try:
                protocol, result = Path(folder) / "p.json", Path(folder) / "r.json"
                protocol.write_text("{}")
                result.write_text(json.dumps({"external_database": "HiRID", "cells": [cell()]}))
                with self.assertRaises(RuntimeError):                             # never scored
                    lockbox_cells(protocol, "HiRID", result)
                frozen_protocol.claim_step(protocol, "continuous_HiRID")
                frozen_protocol.record_step(protocol, "continuous_HiRID", [], [str(result)])
                self.assertEqual(list(lockbox_cells(protocol, "HiRID", result)), ["x@1h"])
                with self.assertRaises(RuntimeError):                             # recorded for another database
                    lockbox_cells(protocol, "SICdb", result)
                result.write_text(json.dumps({"external_database": "HiRID", "cells": []}))
                with self.assertRaises(RuntimeError):                             # changed after scoring
                    lockbox_cells(protocol, "HiRID", result)
            finally:
                frozen_protocol.LEDGER = previous


if __name__ == "__main__":
    unittest.main()
