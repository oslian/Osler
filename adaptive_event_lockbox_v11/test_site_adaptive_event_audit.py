import argparse
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import osler_jepa.frozen_protocol as frozen_protocol
from external_value_conditional_interval_audit import LOCKBOX_DATABASES, external_cohort
from osler_jepa.frozen_protocol import claim_step, record_step, sha256
from site_adaptive_event_audit import PROJECT, form_arguments, loaded_project_files, lockbox_step
from site_adaptive_event_dev import score_stream

DEVELOPMENT = Path("mimiciv_all_49_runtime_cells_development_rescue_audit_20260816.json")
EVENT_CONFIG = Path("site_adaptive_event_amendment_20261028.json")
CELLS = ["hypoxemia_event@1h", "hypoxemia_event@3h"]


class LockboxStepTests(unittest.TestCase):
    """Run from the project folder: the protocol hashes project-relative paths."""

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.ledger = mock.patch.object(frozen_protocol, "LEDGER", self.folder / "ledger")
        self.ledger.start()
        self.addCleanup(self.ledger.stop)
        self.extraction = {h: self.folder / f"site_{h}h.parquet" for h in (1, 3)}
        self.training = {h: self.folder / f"mimic_{h}h.parquet" for h in (1, 3)}
        for path in [*self.extraction.values(), *self.training.values()]:
            path.write_bytes(b"bytes")
        self.previous = self.folder / "previous.json"
        self.previous.write_text("{}", encoding="utf-8")

    def args(self, **changes):
        values = dict(database="SICdb", protocol=self.folder / "protocol.json",
                      mimic_template=str(self.folder / "mimic_{horizon}h.parquet"), development_report=DEVELOPMENT,
                      event_config=EVENT_CONFIG, samples=5, max_rows_per_database=60000,
                      external_template=str(self.folder / "site_{horizon}h.parquet"))
        values.update(changes)
        return argparse.Namespace(**values)

    def write_protocol(self, *, task_types=("binary_event_probability",), drop=None, cells=CELLS, training=(1, 3)):
        args = self.args()
        hashed = (loaded_project_files() | {str(EVENT_CONFIG), str(DEVELOPMENT), str(self.previous)}) - {drop}
        args.protocol.write_text(json.dumps({
            "scored_task_types": list(task_types), "scored_cells": list(cells), "previous_protocol": str(self.previous),
            "frozen_code_sha256": {name: sha256(Path(name)) for name in sorted(hashed)},
            "frozen_training_data_sha256": {str(self.training[h]): sha256(self.training[h]) for h in training},
            "deployment_form_arguments": form_arguments(args)}), encoding="utf-8")
        return args

    def addendum(self, protocol, horizons=(1, 3)):
        claim_step(protocol, "addendum_SICdb")
        record_step(protocol, "addendum_SICdb", [str(self.extraction[h]) for h in horizons], [])

    def ready(self):
        args = self.write_protocol()
        self.addendum(self.previous)
        self.addendum(args.protocol)
        return args

    def test_development_database_needs_no_protocol(self):
        self.assertIsNone(lockbox_step(self.args(database="eICU", protocol=None)))

    def test_lockbox_requires_a_protocol(self):
        with self.assertRaises(ValueError):
            lockbox_step(self.args(protocol=None))

    def test_protocol_must_include_binary_events(self):
        with self.assertRaisesRegex(RuntimeError, "outside this protocol's lockbox scope"):
            lockbox_step(self.write_protocol(task_types=("continuous_value",)))

    def test_scored_cells_must_match_exactly(self):
        with self.assertRaisesRegex(RuntimeError, "scored_cells"):
            lockbox_step(self.write_protocol(cells=CELLS[:1]))

    def test_every_loaded_project_module_must_be_hashed(self):
        with self.assertRaisesRegex(RuntimeError, "not hashed"):
            lockbox_step(self.write_protocol(drop="osler_jepa/online_platt.py"))

    def test_previous_protocol_must_be_hashed(self):
        with self.assertRaisesRegex(RuntimeError, "not hashed"):
            lockbox_step(self.write_protocol(drop=str(self.previous)))

    def test_training_files_must_be_hashed(self):
        with self.assertRaisesRegex(RuntimeError, "training files not hashed"):
            lockbox_step(self.write_protocol(training=(1,)))

    def test_changed_training_file_is_refused(self):
        args = self.write_protocol()
        self.training[3].write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed after protocol freeze"):
            lockbox_step(args)

    def test_invocation_must_match_the_frozen_form(self):
        self.write_protocol()
        with self.assertRaisesRegex(RuntimeError, "invocation differs"):
            lockbox_step(self.args(samples=3))

    def test_must_run_from_the_project_folder(self):
        args = self.write_protocol()
        current = os.getcwd()
        os.chdir(self.folder)
        try:
            with self.assertRaisesRegex(RuntimeError, "project folder"):
                lockbox_step(args)
        finally:
            os.chdir(current)

    def test_project_module_loaded_from_elsewhere_is_refused(self):
        elsewhere = types.ModuleType("lockbox_addendum")
        elsewhere.__file__ = str(self.folder / "other_checkout" / "lockbox_addendum.py")
        with mock.patch.dict(sys.modules, {"lockbox_addendum": elsewhere}):
            with self.assertRaisesRegex(RuntimeError, "not loaded from"):
                loaded_project_files()

    def test_addendum_is_required_and_must_record_both_horizons(self):
        args = self.write_protocol()
        with self.assertRaises(RuntimeError):
            lockbox_step(args)
        self.addendum(args.protocol, horizons=(1,))
        with self.assertRaisesRegex(RuntimeError, "not recorded in the addendum"):
            lockbox_step(args)

    def test_changed_extraction_is_refused(self):
        args = self.ready()
        self.extraction[3].write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed after the addendum"):
            lockbox_step(args)

    def test_extraction_must_match_the_previous_addendum(self):
        args = self.write_protocol()
        self.addendum(self.previous)
        self.extraction[3].write_bytes(b"re-extracted")
        self.addendum(args.protocol)
        with self.assertRaisesRegex(RuntimeError, "differ from the previous protocol's addendum"):
            lockbox_step(args)

    def test_scoring_is_claimed_once_and_unlocks_the_data(self):
        args = self.ready()
        with self.assertRaisesRegex(RuntimeError, "only after a one-shot scoring step is claimed"):
            external_cohort(args, "hypoxemia_event", 6)
        self.assertEqual(lockbox_step(args), "binary_SICdb")
        frame, info = external_cohort(args, "hypoxemia_event", 6)        # past the guard: no 6h file here
        self.assertTrue(frame.empty and info["reason"] == "file_missing")
        with self.assertRaises(FileExistsError):
            lockbox_step(args)


class LockboxDataGuardTests(unittest.TestCase):
    def test_every_evaluator_needs_a_claimed_step_to_read_lockbox_data(self):
        self.assertEqual(LOCKBOX_DATABASES, {"SICdb", "HiRID"})
        folder = Path(tempfile.mkdtemp())
        protocol = folder / "p.json"
        protocol.write_text("{}", encoding="utf-8")
        with mock.patch.object(frozen_protocol, "LEDGER", folder / "ledger"):
            for database in sorted(LOCKBOX_DATABASES):
                for extra in ({}, {"protocol": protocol}, {"protocol": protocol, "claimed_step": f"binary_{database}"}):
                    args = argparse.Namespace(database=database, external_template="x_{horizon}h.parquet", **extra)
                    with self.assertRaisesRegex(RuntimeError, "lockbox"):
                        external_cohort(args, "hypoxemia_event", 1)

    def test_invalid_predictions_fail_the_stream(self):
        probability = np.array([0.1, np.nan, 0.2])
        result = score_stream(None, np.arange(3), np.zeros(3), probability, np.zeros(3), np.full(3, 0.1),
                              axes=("patient",), seed=0)
        self.assertEqual((result["invalid_predictions"], result["full_pass"]), (1, False))
        self.assertEqual(Path.cwd().resolve(), PROJECT)


if __name__ == "__main__":
    unittest.main()
