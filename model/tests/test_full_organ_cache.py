import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.audit_reconstruction_raw_state import summarize
from scripts.cache_full_organ_compartments import (
    append_batch, capture_native, checked_organs, open_cache, pack_organs,
    payload_digest, unpack_organs, validate_cache,
)
from scripts.cache_nonlinear_probe import encode_features
from whole_body_joint_jepa import JointWholeBodyJEPAWorldModel
from whole_body_jepa.history_event_input import expand_history_projection
from whole_body_jepa.native_body_reconstruction import NativeBodyReconstruction
from whole_body_jepa.native_organ_state_residual import NativeOrganStateResidual


class FullOrganCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ref = self.root/'reference.sqlite'
        with closing(sqlite3.connect(self.ref)) as db, db:
            db.execute('CREATE TABLE features (patient TEXT,target INTEGER,payload BLOB,digest TEXT,PRIMARY KEY(patient,target))')
            for patient, target in [('a', 0), ('a', 1), ('b', 0)]:
                payload, digest = encode_features(np.zeros(64), np.zeros(265), float(target), 0.)
                db.execute('INSERT INTO features VALUES (?,?,?,?)', (patient, target, payload, digest))
        self.contract = dict(role='fit', organ_shape=[2, 3])
        self.roles = dict(a='fit', b='fit')
        self.path = self.root/'organs.sqlite'

    def open(self, contract=None, roles=None):
        return open_cache(self.path, self.ref, contract or self.contract, roles or self.roles)

    @staticmethod
    def extract(patient, target, reference):
        return np.full((2, 3), target, np.float32), 0., 0.

    def test_partial_resume_and_complete_exact_grid(self):
        db = self.open()
        added, state = append_batch(db, (2, 3), self.extract, 2, lambda: None)
        self.assertEqual(added, 2)
        self.assertEqual(state, dict(expected=3, stored=2, remaining=1, complete=False))
        db.close()
        db = self.open()
        added, state = append_batch(db, (2, 3), self.extract, 50, lambda: None)
        self.assertEqual(added, 1)
        self.assertTrue(state['complete'])
        self.assertEqual(append_batch(db, (2, 3), self.extract, 50, lambda: None)[0], 0)
        db.close()

    def test_contract_and_role_changes_fail_closed(self):
        self.open().close()
        with self.assertRaisesRegex(ValueError, 'contract'):
            self.open(dict(self.contract, source='changed'))
        with self.assertRaisesRegex(ValueError, 'role'):
            self.open(roles=dict(a='selection', b='fit'))

    def test_payload_binds_identity_reference_and_shape(self):
        payload = pack_organs(np.zeros((2, 3), np.float32))
        digest = payload_digest('a', 0, 'reference', payload)
        unpack_organs('a', 0, 'reference', payload, digest, (2, 3))
        for patient, target, ref, shape in [('b', 0, 'reference', (2, 3)),
                ('a', 1, 'reference', (2, 3)), ('a', 0, 'changed', (2, 3)), ('a', 0, 'reference', (3, 2))]:
            with self.subTest(patient=patient, target=target, reference=ref), self.assertRaises(ValueError):
                unpack_organs(patient, target, ref, payload, digest, shape)
        with self.assertRaises(ValueError):
            pack_organs(np.full((2, 3), np.nan, np.float32))

    def test_resume_detects_corrupt_payload(self):
        db = self.open()
        append_batch(db, (2, 3), self.extract, 2, lambda: None)
        db.execute("UPDATE compartments SET payload=x'00' WHERE target=0")
        db.commit(); db.close()
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            self.open()

    def test_changed_reference_digest_rejected(self):
        db = self.open()
        append_batch(db, (2, 3), self.extract, 1, lambda: None)
        db.execute("UPDATE compartments SET reference_digest='changed'")
        db.commit()
        with self.assertRaisesRegex(ValueError, 'reference query'):
            validate_cache(db, (2, 3))
        db.close()

    def test_bad_parity_not_committed(self):
        db = self.open()
        def bad(*args):
            return np.zeros((2, 3), np.float32), 1., 0.
        with self.assertRaisesRegex(ValueError, 'parity'):
            append_batch(db, (2, 3), bad, 1, lambda: None)
        self.assertEqual(validate_cache(db, (2, 3))['stored'], 0)
        db.close()

    def test_disk_guard_prevents_work(self):
        db = self.open()
        def no_space():
            raise RuntimeError('no space')
        with self.assertRaisesRegex(RuntimeError, 'no space'):
            append_batch(db, (2, 3), self.extract, 2, no_space)
        self.assertFalse(validate_cache(db, (2, 3))['complete'])
        db.close()

    def test_failed_batch_preserves_committed_prefix_and_resumes(self):
        with closing(sqlite3.connect(self.ref)) as ref, ref:
            payload, digest = encode_features(np.zeros(64), np.zeros(265), 0., 0.)
            for i in range(260):
                patient = f'synthetic-{i:03}'
                self.roles[patient] = 'fit'
                ref.execute('INSERT INTO features VALUES (?,?,?,?)', (patient, 0, payload, digest))
        db = self.open()
        calls = 0
        def interrupted(*args):
            nonlocal calls
            calls += 1
            if calls == 130:
                raise RuntimeError('synthetic interruption')
            return self.extract(*args)
        with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
            append_batch(db, (2, 3), interrupted, 300, lambda: None)
        self.assertEqual(validate_cache(db, (2, 3))['stored'], 128)
        db.close()
        db = self.open()
        added, status = append_batch(db, (2, 3), self.extract, 300, lambda: None)
        self.assertEqual(added, 135)
        self.assertTrue(status['complete'])
        db.close()

    def test_reference_is_read_only_and_zero_budget_rejected(self):
        db = self.open()
        with self.assertRaises(sqlite3.OperationalError):
            db.execute('DELETE FROM reference.features')
        with self.assertRaises(ValueError):
            append_batch(db, (2, 3), self.extract, 0, lambda: None)
        self.assertEqual(validate_cache(db, (2, 3))['expected'], 3)
        db.close()


class NativeCaptureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        jepa = JointWholeBodyJEPAWorldModel(3, 14, 3, hidden=12, latent=8, token_dim=8,
            variable_module_ids=[0, 1, 2], horizon_values=(3.,), target_encoder_mode='fast')
        expand_history_projection(jepa, 6)
        jepa.belief_input_mode = 'observed_steps'
        self.variables = ('hemoglobin', 'hematocrit', 'glucose')
        self.model = NativeBodyReconstruction(NativeOrganStateResidual(jepa, self.variables, ('hemoglobin',))).eval()
        tokens = torch.zeros(2, 20)
        tokens[:, 3:6] = 1
        self.example = dict(tokens=tokens, event_hours=torch.tensor([0., 1.]), anchor_hour=2., truth_normalized=0.)

    def test_native_body_point_and_reference_parity(self):
        body, organs, prediction = capture_native(self.model, self.example, 'hemoglobin')
        with torch.no_grad():
            expected = float(self.model(self.example['tokens'], self.example['event_hours'], 2., 'hemoglobin')[0])
            from_body = float(self.model.readout(torch.from_numpy(body)[None])[0, 0])
        self.assertEqual(prediction, expected)
        self.assertEqual(prediction, from_body)
        raw, _ = summarize(self.example['tokens'], self.example['event_hours'], 2., self.variables, 'hemoglobin')
        reference = dict(body=body, raw=raw, truth=0., prediction=prediction)
        replay, b, p = checked_organs(self.model, self.example, 'hemoglobin', reference)
        np.testing.assert_array_equal(replay, organs)
        self.assertEqual((b, p), (0., 0.))
        with self.assertRaisesRegex(ValueError, 'truth'):
            checked_organs(self.model, self.example, 'hemoglobin', dict(reference, truth=1.))

    def test_target_and_sibling_hidden_from_all_compartments(self):
        before = capture_native(self.model, self.example, 'hemoglobin')
        tokens = self.example['tokens'].clone()
        tokens[:, :2] = torch.tensor([100., -100.])
        tokens[:, 6:8] = 999.
        after = capture_native(self.model, dict(self.example, tokens=tokens), 'hemoglobin')
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_invalid_time_removes_hook_and_zero_is_observed(self):
        count = len(self.model.trajectory._forward_hooks)
        for times in (torch.tensor([0., 3.]), torch.tensor([1., 0.]), torch.tensor([0., float('nan')])):
            with self.assertRaises(ValueError):
                capture_native(self.model, dict(self.example, event_hours=times), 'hemoglobin')
            self.assertEqual(len(self.model.trajectory._forward_hooks), count)
        before = capture_native(self.model, self.example, 'hemoglobin')[0]
        tokens = self.example['tokens'].clone()
        tokens[0, 5] = 0
        after = capture_native(self.model, dict(self.example, tokens=tokens), 'hemoglobin')[0]
        self.assertFalse(np.array_equal(before, after))


if __name__ == '__main__':
    unittest.main()
