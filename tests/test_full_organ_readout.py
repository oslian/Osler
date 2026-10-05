from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import torch

from scripts.cache_full_organ_compartments import pack_organs, payload_digest
from scripts.cache_nonlinear_probe import encode_features
from scripts.train_full_organ_readout import (
    KINDS, PARAMETERS, features, fit_readers, paired_data, require_complete_grid,
    require_model, verify_raw,
)
from scripts import train_full_organ_readout as reader
from whole_body_jepa.multisource_intervals import sha256


class FullOrganReadoutTest(unittest.TestCase):
    @staticmethod
    def reports():
        variables = [f'v{i}' for i in range(53)]
        return {(f, r): dict(contract=dict(fold=f, role=r, source_fold=0 if f == 2 else f,
            variables=variables.copy(), organ_shape=[2, 3], organ_names=['a', 'b']),
            complete=True, stored=94, expected=94, remaining=0,
            support={v: dict(stored=2, expected=2) for v in variables[:47]})
            for f in range(5) for r in ('fit', 'selection')}

    @staticmethod
    def data(n, offset=0):
        rng = np.random.default_rng(12+offset)
        return dict(body=rng.normal(size=(n, 64)), organs=rng.normal(size=(n, 6)),
                    raw=rng.normal(size=(n, 265)), truth=rng.normal(size=n), prediction=np.zeros(n))

    def test_all_ten_roles_and_inventory_required(self):
        reports = self.reports()
        variables, shape = require_complete_grid(reports)
        self.assertEqual((len(variables), shape), (53, (2, 3)))
        del reports[4, 'selection']
        with self.assertRaisesRegex(ValueError, 'ten'):
            require_complete_grid(reports)

    def test_false_completion_partial_support_and_wrong_source_rejected(self):
        changes = [dict(complete=False), dict(stored=93), dict(remaining=1), dict(expected=93)]
        for change in changes:
            reports = self.reports(); reports[0, 'fit'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                require_complete_grid(reports)
        reports = self.reports(); reports[2, 'fit']['contract']['source_fold'] = 2
        with self.assertRaisesRegex(ValueError, 'source fold'):
            require_complete_grid(reports)
        reports = self.reports(); reports[4, 'selection']['support']['v0']['stored'] = 1
        with self.assertRaisesRegex(ValueError, 'support'):
            require_complete_grid(reports)

    def test_schema_and_family_mismatch_rejected(self):
        reports = self.reports(); reports[3, 'fit']['contract']['organ_names'] = ['b', 'a']
        with self.assertRaisesRegex(ValueError, 'schema'):
            require_complete_grid(reports)
        reports = self.reports()
        reports[4, 'fit']['support']['v52'] = reports[4, 'fit']['support'].pop('v0')
        with self.assertRaisesRegex(ValueError, 'family'):
            require_complete_grid(reports)

    def test_same_process_and_equal_width_controls(self):
        data, ids = self.data(4), ['a', 'b', 'c', 'd']
        projection = np.random.default_rng(20261003).normal(size=(64, 6))/8
        xs = {k: features(data, ids, k, projection) for k in KINDS}
        self.assertEqual(xs['body'].shape, (4, 223))
        for kind in ('organs', 'shuffled', 'expanded'):
            self.assertEqual(xs[kind].shape, (4, 229))
            np.testing.assert_array_equal(xs[kind][:, :64], data['body'])
            np.testing.assert_array_equal(xs[kind][:, -159:], data['raw'][:, 106:])
        np.testing.assert_array_equal(xs['shuffled'][:, 64:70], np.roll(data['organs'], 1, axis=0))
        np.testing.assert_array_equal(xs['expanded'][:, 64:70], data['body']@projection)
        changed = deepcopy(data); changed['raw'][:, :106] = 999
        np.testing.assert_array_equal(features(changed, ids, 'organs', projection), xs['organs'])

    def test_bad_feature_identity_and_nonfinite_rejected(self):
        d, p = self.data(3), np.zeros((64, 6))
        for ids in (['a', 'a', 'c'], ['c', 'b', 'a'], ['a']):
            with self.assertRaises(ValueError):
                features(d, ids, 'shuffled', p)
        d['organs'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            features(d, ['a', 'b', 'c'], 'organs', p)

    def test_exact_organ_reference_pairing_and_zero(self):
        with closing(sqlite3.connect(':memory:')) as ref, closing(sqlite3.connect(':memory:')) as org:
            ref.execute('CREATE TABLE features(patient TEXT,target INT,payload BLOB,digest TEXT)')
            org.execute('CREATE TABLE compartments(patient TEXT,target INT,reference_digest TEXT,payload BLOB,digest TEXT)')
            for p in ('a', 'b'):
                payload, digest = encode_features(np.zeros(64), np.zeros(265), 0., 0.)
                ref.execute('INSERT INTO features VALUES (?,?,?,?)', (p, 0, payload, digest))
                organs = pack_organs(np.zeros((2, 3), np.float32))
                org.execute('INSERT INTO compartments VALUES (?,?,?,?,?)',
                    (p, 0, digest, organs, payload_digest(p, 0, digest, organs)))
            ids, data = paired_data(ref, org, 0, (2, 3), dict(a='fit', b='fit'), 'fit')
            self.assertEqual(ids, ['a', 'b'])
            np.testing.assert_array_equal(data['organs'], np.zeros((2, 6)))
            np.testing.assert_array_equal(data['truth'], np.zeros(2))
            with self.assertRaisesRegex(ValueError, 'role'):
                paired_data(ref, org, 0, (2, 3), dict(a='selection', b='fit'), 'fit')
            org.execute("UPDATE compartments SET reference_digest='changed' WHERE patient='a'")
            with self.assertRaisesRegex(ValueError, 'binding'):
                paired_data(ref, org, 0, (2, 3), dict(a='fit', b='fit'), 'fit')

    def test_fit_four_full_spec_readers_and_prediction_replay(self):
        fit, query = self.data(80), self.data(8, 1)
        fids, qids = [f'f{i:03}' for i in range(80)], [f'q{i:03}' for i in range(8)]
        projection = np.random.default_rng(20261003).normal(size=(64, 6))/8
        models, predictions, diagnostics = fit_readers(fit, query, fids, qids, np.ones(80), projection)
        self.assertEqual(set(models), set(KINDS))
        with threadpool_limits(limits=4):
            for kind, model in models.items():
                self.assertEqual(model.n_iter_, 100)
                np.testing.assert_array_equal(model.predict(features(query, qids, kind, projection)), predictions[kind])
                self.assertGreaterEqual(diagnostics[kind]['mean_leaves'], 1)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            fit_readers(fit, query, fids, fids[:8], np.ones(80), projection)

    def test_raw_parity_parameters_support_and_roles(self):
        f, q = self.data(80), self.data(8, 1)
        fi, qi = [f'f{i:03}' for i in range(80)], [f'q{i:03}' for i in range(8)]
        roles = {**dict.fromkeys(fi, 'fit'), **dict.fromkeys(qi, 'selection')}
        components = dict.fromkeys(fi+qi, 'source')
        with threadpool_limits(limits=4):
            model = HistGradientBoostingRegressor(**PARAMETERS).fit(f['raw'], f['truth'], sample_weight=np.ones(80))
            saved = pd.DataFrame(dict(patient=qi, component='source', target='v0', fold=0,
                                     truth=q['truth'], native=q['prediction'], raw_probe=model.predict(q['raw'])))
            bundle = dict(target='v0', parameters=PARAMETERS, models=dict(raw_probe=model))
            meta = dict(fold=0, target='v0', fit_patients=80, selection_patients=8)
            _, prediction, weights = verify_raw(bundle, saved, meta, 0, 'v0', fi, qi, f, q, roles, components)
            np.testing.assert_array_equal(prediction, saved.raw_probe)
            np.testing.assert_array_equal(weights, np.ones(80))
            with self.assertRaisesRegex(ValueError, 'support'):
                verify_raw(bundle, saved, dict(meta, fit_patients=79), 0, 'v0', fi, qi, f, q, roles, components)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                verify_raw(bundle, saved, meta, 0, 'v0', fi, qi, f, q, dict(roles, f000='selection'), components)
            broken = saved.copy(); broken.loc[0, 'raw_probe'] += 1.
            with self.assertRaises(AssertionError):
                verify_raw(bundle, broken, meta, 0, 'v0', fi, qi, f, q, roles, components)
            model.set_params(max_leaf_nodes=31)
            with self.assertRaisesRegex(ValueError, 'parameters'):
                require_model(model, 265)

    def test_training_entry_saves_all_arms_and_inventory(self):
        # Only the expensive ten-cache preflight is replaced by synthetic input;
        # real SQLite joins, contracts, HGB fits, raw replay and saving execute.
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=4):
            root = Path(temp); dest = root/'organs'; dest.mkdir()
            artifacts = root/'artifacts'; artifacts.mkdir()
            rawroot = root/'nonlinear_probe_20261001'; rawroot.mkdir()
            rawdir = rawroot/'fold0_probes'; rawdir.mkdir()
            variables = [f'v{i}' for i in range(53)]
            fit, query = self.data(80), self.data(8, 1)
            fids, qids = [f'f{i:03}' for i in range(80)], [f'q{i:03}' for i in range(8)]
            roles = {**dict.fromkeys(fids, 'fit'), **dict.fromkeys(qids, 'selection')}
            fp, provenance = root/'partitions_20260922.pt', root/'provenance.parquet'
            torch.save(dict(folds={0: roles}), fp)
            pd.DataFrame(dict(uniquepid=fids+qids, component='source')).to_parquet(provenance, index=False)
            grid, bindings = {}, {}
            for role, ids, data in [('fit', fids, fit), ('selection', qids, query)]:
                ref = rawroot/f'fold0_{role}.sqlite'; organ = dest/f'fold0_{role}.sqlite'
                with closing(sqlite3.connect(ref)) as rdb, closing(sqlite3.connect(organ)) as odb:
                    rdb.execute('CREATE TABLE features(patient TEXT,target INT,payload BLOB,digest TEXT)')
                    odb.execute('CREATE TABLE compartments(patient TEXT,target INT,reference_digest TEXT,payload BLOB,digest TEXT)')
                    for i, patient in enumerate(ids):
                        payload, digest = encode_features(data['body'][i], data['raw'][i], data['truth'][i], data['prediction'][i])
                        rdb.execute('INSERT INTO features VALUES (?,?,?,?)', (patient, 0, payload, digest))
                        op = pack_organs(data['organs'][i].astype(np.float32).reshape(2, 3))
                        odb.execute('INSERT INTO compartments VALUES (?,?,?,?,?)',
                            (patient, 0, digest, op, payload_digest(patient, 0, digest, op)))
                    rdb.commit(); odb.commit()
                bindings[str(ref)] = sha256(ref)
                grid[0, role] = dict(support={'v0': dict(expected=len(ids), stored=len(ids))},
                                    contract=dict(bindings={str(ref): sha256(ref)}))
            for path in (fp, provenance, Path('scripts/bounded_site_sampling.py')):
                bindings[str(path)] = sha256(path)
            contract = dict(fold=0, parameters=PARAMETERS, bindings=bindings)
            cp = rawdir/'contract.json'; cp.write_text(json.dumps(contract))
            raw = HistGradientBoostingRegressor(**PARAMETERS).fit(fit['raw'], fit['truth'], sample_weight=np.ones(80))
            mp, pp = rawdir/'v0.joblib', rawdir/'v0.parquet'
            joblib.dump(dict(target='v0', parameters=PARAMETERS, models=dict(raw_probe=raw)), mp)
            pd.DataFrame(dict(patient=qids, component='source', target='v0', fold=0,
                truth=query['truth'], native=query['prediction'], raw_probe=raw.predict(query['raw']))).to_parquet(pp, index=False)
            meta = dict(fold=0, target='v0', fit_patients=80, selection_patients=8,
                        contract_sha256=sha256(cp), model_sha256=sha256(mp), prediction_sha256=sha256(pp))
            (rawdir/'v0.json').write_text(json.dumps(meta))
            report = dict(fold=0, parameters=PARAMETERS, bindings=bindings, targets=dict(v0=meta))
            (artifacts/'native_reconstruction_nonlinear_training_fold0_20261001.json').write_text(json.dumps(report))
            protocol = root/'protocol.md'; protocol.write_text('synthetic orchestration fixture')
            with patch.multiple(reader, ROOT=root, DEST=dest, READERS=dest/'readers', ARTIFACTS=artifacts, PROTOCOL=protocol), \
                    patch.object(reader, 'verify_full_grid', return_value=(grid, variables, (2, 3), bindings.copy())):
                reader.run_fold(0)
                path = artifacts/'native_reconstruction_full_organ_readers_fold0_20261003.json'
                result = json.loads(path.read_text())
                self.assertTrue(result['complete'])
                self.assertFalse(result['promotion_allowed'])
                self.assertEqual(len(result['targets']), 53)
                self.assertEqual(result['targets']['v52']['status'], 'unsupported')
                self.assertEqual(result['targets']['v0']['fit_patients'], 80)
                bundle = joblib.load(dest/'readers/fold0/v0.joblib')
                saved = pd.read_parquet(dest/'readers/fold0/v0.parquet')
                self.assertEqual(set(bundle['models']), set(KINDS))
                np.testing.assert_array_equal(saved.raw, raw.predict(query['raw']))
                with self.assertRaisesRegex(ValueError, 'already exist'):
                    reader.run_fold(0)


if __name__ == '__main__':
    unittest.main()
